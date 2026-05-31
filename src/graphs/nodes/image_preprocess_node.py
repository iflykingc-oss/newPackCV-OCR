# -*- coding: utf-8 -*-
"""
图像预处理节点 - 稳定版
用于OCR前的图像增强处理
"""
import os
import json
import time
import logging
import requests
from typing import Optional, Dict, Any, List
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
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        pil_img = Image.open(BytesIO(resp.content))
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        logger.warning(f"图片下载失败: {e}")
        return None


def preprocess_image(img: np.ndarray) -> np.ndarray:
    """
    OCR前图像预处理
    1. 灰度化
    2. CLAHE对比度增强
    3. 轻度锐化
    """
    # 灰度化
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # CLAHE增强
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 轻度高斯去噪
    denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)

    # 转回BGR
    result = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
    return result


def image_preprocess_node(
    state: ImagePreprocessInput,
    config: RunnableConfig,
    runtime: Runtime[Context]
) -> ImagePreprocessOutput:
    """
    title: 图像预处理
    desc: 对输入图片进行OCR前预处理，包括灰度化、CLAHE对比度增强、轻度去噪。
          处理后的图片上传到对象存储供后续OCR节点使用。
    integrations: 对象存储
    """
    ctx = runtime.context
    start_time = time.time()

    logger.info(f"开始图像预处理: {state.package_image.url if state.package_image else 'no image'}")

    try:
        # 下载图片
        img_url = ""
        if state.package_image and state.package_image.url:
            img_url = state.package_image.url

        if not img_url:
            logger.error("无可用图片URL")
            return ImagePreprocessOutput(
                preprocessed_image=None,
                is_enhanced=False
            )

        img = download_image(img_url)
        if img is None:
            logger.error("图片下载失败")
            return ImagePreprocessOutput(
                preprocessed_image=None,
                is_enhanced=False
            )

        logger.info(f"图片下载成功: {img.shape}")

        # 预处理
        preprocessed = preprocess_image(img)
        logger.info(f"预处理完成: {preprocessed.shape}")

        # 保存为PNG并上传
        success, img_encoded = cv2.imencode('.png', preprocessed)
        if not success:
            logger.error("图片编码失败")
            return ImagePreprocessOutput(
                preprocessed_image=None,
                is_enhanced=False
            )

        img_bytes = img_encoded.tobytes()

        # 上传S3
        storage = _get_s3_storage()
        file_name = f"preprocessed/preprocessed_{int(time.time())}.png"
        key = storage.upload_file(
            file_content=img_bytes,
            file_name=file_name,
            content_type='image/png'
        )
        presigned_url = storage.generate_presigned_url(key=key, expire_time=3600)
        logger.info(f"预处理图片已上传: {presigned_url[:60]}...")

        elapsed_time = time.time() - start_time
        logger.info(f"图像预处理完成，耗时: {elapsed_time:.2f}秒")

        return ImagePreprocessOutput(
            preprocessed_image=File(url=presigned_url, file_type="image"),
            is_enhanced=True
        )

    except Exception as e:
        logger.error(f"图像预处理异常: {e}", exc_info=True)
        return ImagePreprocessOutput(
            preprocessed_image=None,
            is_enhanced=False
        )
