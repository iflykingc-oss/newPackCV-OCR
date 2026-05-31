# -*- coding: utf-8 -*-
"""
图像预处理节点 - 深度优化版 V2.3
RapidOCR自带文本检测+方向分类+识别，预处理节点只做轻量级增强
1. 下载图片 → CLAHE对比度增强 + 轻度锐化 → 上传S3
2. 如果增强失败，直接传递原始图片URL
"""
import os
import time
import logging
import requests
from typing import Optional, Dict, Any, Tuple
from io import BytesIO

import cv2
import numpy as np
from PIL import Image

from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context
from graphs.state import ImagePreprocessInput, ImagePreprocessOutput
from utils.file.file import File
from coze_coding_dev_sdk.s3 import S3SyncStorage

logger = logging.getLogger(__name__)


def _get_s3_storage() -> S3SyncStorage:
    """获取S3对象存储客户端"""
    return S3SyncStorage(
        endpoint_url=os.getenv("COZE_BUCKET_ENDPOINT_URL"),
        access_key="",
        secret_key="",
        bucket_name=os.getenv("COZE_BUCKET_NAME"),
        region="cn-beijing",
    )


def download_image(url: str) -> Optional[np.ndarray]:
    """下载图片并转为OpenCV格式"""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        resp = requests.get(url, timeout=30, headers=headers)
        resp.raise_for_status()
        pil_img = Image.open(BytesIO(resp.content))
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        logger.warning(f"图片下载失败: {e}")
        return None


def enhance_for_ocr(img: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    轻量级OCR预处理管线
    RapidOCR内置文本检测+方向分类，预处理只需提供高质量输入
    """
    info: Dict[str, Any] = {
        "original_size": img.shape[:2] if img is not None else (0, 0),
        "enhanced": False,
        "resized": False
    }

    if img is None or img.size == 0:
        return img, info

    result = img.copy()
    h, w = result.shape[:2]

    # 1. 大图智能缩放（RapidOCR推荐最大边不超过2000px）
    max_dimension = 2000
    if max(h, w) > max_dimension:
        scale = max_dimension / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        result = cv2.resize(result, (new_w, new_h), interpolation=cv2.INTER_AREA)
        info["resized"] = True
        info["resize_scale"] = round(scale, 3)
        logger.info(f"大图缩放: {w}x{h} -> {new_w}x{new_h}")

    # 2. CLAHE对比度增强（Lab空间处理亮度通道）
    try:
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l_channel)
        lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
        result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        info["enhanced"] = True
    except Exception as e:
        logger.warning(f"CLAHE增强失败: {e}")

    # 3. 轻度锐化（Unsharp Masking）
    try:
        gaussian = cv2.GaussianBlur(result, (0, 0), 2.0)
        result = cv2.addWeighted(result, 1.5, gaussian, -0.5, 0)
    except Exception as e:
        logger.warning(f"锐化失败: {e}")

    info["output_size"] = result.shape[:2]
    return result, info


def image_preprocess_node(
    state: ImagePreprocessInput,
    config: RunnableConfig,
    runtime: Runtime[Context]
) -> ImagePreprocessOutput:
    """
    title: 图像预处理
    desc: 对输入图片进行OCR前轻量级预处理（CLAHE对比度增强+锐化），处理后的图片上传到对象存储。
          RapidOCR自带文本检测+方向分类，无需额外方向/区域检测。
    integrations: 对象存储
    """
    ctx = runtime.context
    start_time = time.time()

    try:
        img_url = ""
        if state.package_image and state.package_image.url:
            img_url = state.package_image.url

        if not img_url:
            logger.error("无可用图片URL")
            return ImagePreprocessOutput(
                is_enhanced=False
            )

        img = download_image(img_url)
        if img is None:
            logger.error("图片下载失败")
            return ImagePreprocessOutput(
                is_enhanced=False
            )

        logger.info(f"图片下载成功: shape={img.shape}")

        # 轻量级图像增强
        enhanced_img, processing_info = enhance_for_ocr(img)
        logger.info(f"图像增强完成: {processing_info}")

        # 保存增强后的图片并上传S3（使用JPEG格式减小文件体积）
        success, img_encoded = cv2.imencode('.jpg', enhanced_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not success:
            logger.error("图片编码失败")
            return ImagePreprocessOutput(
                is_enhanced=False
            )

        img_bytes = img_encoded.tobytes()

        # 上传S3
        storage = _get_s3_storage()
        file_name = f"preprocessed/preprocessed_{int(time.time())}.jpg"
        key = storage.upload_file(
            file_content=img_bytes,
            file_name=file_name,
            content_type='image/jpeg'
        )
        presigned_url = storage.generate_presigned_url(key=key, expire_time=3600)
        logger.info(f"预处理图片已上传: {presigned_url[:60]}...")

        elapsed_time = time.time() - start_time
        processing_info["elapsed_time"] = round(elapsed_time, 2)
        logger.info(f"图像预处理完成，耗时: {elapsed_time:.2f}秒")

        is_rotated = processing_info.get("rotation_angle", 0) != 0

        return ImagePreprocessOutput(
            preprocessed_image=File(url=presigned_url, file_type="image"),
            is_rotated=is_rotated,
            is_enhanced=processing_info.get("enhanced", False),
            processing_info=processing_info
        )

    except Exception as e:
        logger.error(f"图像预处理异常: {e}", exc_info=True)
        return ImagePreprocessOutput(
            is_enhanced=False
        )
