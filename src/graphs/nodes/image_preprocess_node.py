# -*- coding: utf-8 -*-
"""
图片预处理节点
针对瓶子等包装的图像增强、去噪、校正等预处理操作
"""

import os
import time
import traceback
import requests
from typing import Dict, Any
from datetime import datetime
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context
from coze_coding_dev_sdk.s3 import S3SyncStorage

import cv2
import numpy as np

from graphs.state import ImagePreprocessInput, ImagePreprocessOutput
from utils.file.file import File, FileOps


def _upload_image_to_storage(image_array: np.ndarray, file_name: str) -> str:
    """上传图片到对象存储，返回签名URL"""
    try:
        is_success, buffer = cv2.imencode('.jpg', image_array, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not is_success:
            raise Exception("图片编码失败")

        storage = S3SyncStorage(
            endpoint_url=os.getenv("COZE_BUCKET_ENDPOINT_URL"),
            access_key="",
            secret_key="",
            bucket_name=os.getenv("COZE_BUCKET_NAME"),
            region="cn-beijing",
        )
        image_bytes = buffer.tobytes()
        key = storage.upload_file(
            file_content=image_bytes,
            file_name=file_name,
            content_type='image/jpeg'
        )
        url = storage.generate_presigned_url(key=key, expire_time=86400)
        return url
    except Exception as e:
        print(f"[图片预处理] 上传图片失败: {e}")
        raise


def image_preprocess_node(
    state: ImagePreprocessInput,
    config: RunnableConfig,
    runtime: Runtime[Context]
) -> ImagePreprocessOutput:
    """
    title: 图片预处理
    desc: 对包装图片进行增强、去噪、校正等预处理，提升OCR识别准确率
    integrations: OpenCV
    """
    ctx = runtime.context
    
    # 获取图片路径
    image_url = state.package_image.url
    
    # 如果是URL，先下载到临时目录
    local_path = None
    if image_url.startswith("http://") or image_url.startswith("https://"):
        temp_path = f"/tmp/preprocess_input_{int(time.time())}.jpg"
        try:
            resp = requests.get(image_url, timeout=30)
            resp.raise_for_status()
            with open(temp_path, "wb") as f:
                f.write(resp.content)
            local_path = temp_path
        except Exception as e:
            print(f"[图片预处理] 下载图片失败: {e}，使用原URL")
            local_path = image_url
    else:
        local_path = image_url
    
    # 使用OpenCV进行图像预处理
    try:
        # 读取图片
        img = cv2.imread(local_path)
        if img is None:
            raise Exception(f"无法读取图片: {local_path}")
        
        processing_info = {}
        is_rotated = False
        is_enhanced = False
        
        # 1. 图像增强（提升对比度、亮度）- 使用CLAHE
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        is_enhanced = True
        processing_info["enhanced"] = True
        
        # 2. 去噪（针对瓶子包装的噪点）
        denoised = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
        processing_info["denoised"] = True
        
        # 3. 边缘增强（针对小字体喷码）
        kernel_sharpen = np.array([[-1, -1, -1],
                                   [-1,  9, -1],
                                   [-1, -1, -1]])
        sharpened = cv2.filter2D(denoised, -1, kernel_sharpen)
        processing_info["sharpened"] = True
        
        # 注意：不再将图像转为二值图，PaddleOCR等引擎需要彩色或灰度图
        # 保存增强后的彩色图（而非二值图），以获得更好的OCR效果
        processed_image = sharpened
        
        # 上传到对象存储
        file_name = f"preprocessed/preprocess_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        processed_url = _upload_image_to_storage(processed_image, file_name)
        
        # 创建File对象
        processed_file = File(url=processed_url, file_type="image")
        
        processing_info["original_size"] = f"{img.shape[1]}x{img.shape[0]}"
        processing_info["output_size"] = f"{processed_image.shape[1]}x{processed_image.shape[0]}"
        processing_info["processing_steps"] = ["enhance", "denoise", "sharpen"]
        
        print(f"[图片预处理] 完成: {processing_info}")
        
        return ImagePreprocessOutput(
            preprocessed_image=processed_file,
            is_rotated=is_rotated,
            is_enhanced=is_enhanced,
            processing_info=processing_info
        )
        
    except Exception as e:
        # 如果OpenCV不可用或处理失败，返回原图
        print(f"[图片预处理] 失败，使用原图: {e}")
        traceback.print_exc()
        return ImagePreprocessOutput(
            preprocessed_image=state.package_image,
            is_rotated=False,
            is_enhanced=False,
            processing_info={"error": str(e), "fallback": "use_original"}
        )
