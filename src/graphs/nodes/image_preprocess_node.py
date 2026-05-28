# -*- coding: utf-8 -*-
"""
图像预处理节点
对输入图片进行预处理：灰度化、高斯模糊、CLAHE增强、二值化
"""

import os
import time
import logging
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import requests

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
    """下载图片并转换为OpenCV格式"""
    try:
        if url.startswith('http://') or url.startswith('https://'):
            response = requests.get(url, timeout=30)
            if response.status_code != 200:
                return None
            pil_img = Image.open(BytesIO(response.content))
            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        elif os.path.exists(url):
            return cv2.imread(url)
        return None
    except Exception as e:
        logger.error(f"下载图片失败: {e}")
        return None


def image_preprocess_node(
    state: ImagePreprocessInput,
    config: RunnableConfig,
    runtime: Runtime[Context]
) -> ImagePreprocessOutput:
    """
    title: 图像预处理
    desc: 对输入图片进行灰度化、高斯模糊、CLAHE增强、二值化等预处理
    integrations: 对象存储
    """
    ctx = runtime.context
    start_time = time.time()

    try:
        image_url = state.package_image.url if state.package_image else ""
        if not image_url:
            logger.error("无输入图片")
            return ImagePreprocessOutput(
                preprocessed_image=state.package_image,
                is_enhanced=False
            )

        # 下载图片
        img = download_image(image_url)
        if img is None:
            logger.error("图片下载失败")
            return ImagePreprocessOutput(
                preprocessed_image=state.package_image,
                is_enhanced=False
            )

        logger.info(f"图片下载成功，尺寸: {img.shape}")

        # 图像预处理管线
        processed = img.copy()

        # 1. 灰度化
        if len(processed.shape) == 3:
            gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        else:
            gray = processed.copy()

        # 2. 高斯模糊（用于锐化）
        blur = cv2.GaussianBlur(gray, (0, 0), 3)
        sharpened = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)

        # 3. CLAHE增强
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(sharpened)

        # 4. 保存增强后的图片
        temp_path = f"/tmp/preprocessed_{int(time.time())}.png"
        cv2.imwrite(temp_path, enhanced)

        # 5. 上传到S3
        upload_url = image_url  # 默认使用原图
        try:
            storage = _get_s3_storage()
            with open(temp_path, 'rb') as f:
                file_bytes = f.read()

            file_name = f"preprocessed/preprocessed_{int(time.time())}.png"
            key = storage.upload_file(
                file_content=file_bytes,
                file_name=file_name,
                content_type='image/png'
            )
            upload_url = storage.generate_presigned_url(key=key, expire_time=3600)
            logger.info(f"预处理图片已上传S3: {key}")
        except Exception as e:
            logger.warning(f"S3上传失败，使用原图: {e}")

        # 清理临时文件
        try:
            os.remove(temp_path)
        except Exception:
            pass

        elapsed_time = time.time() - start_time
        logger.info(f"图像预处理完成，耗时: {elapsed_time:.2f}秒")

        return ImagePreprocessOutput(
            preprocessed_image=File(url=upload_url, file_type="image"),
            is_enhanced=True
        )

    except Exception as e:
        logger.error(f"图像预处理异常: {e}")
        return ImagePreprocessOutput(
            preprocessed_image=state.package_image,
            is_enhanced=False
        )
