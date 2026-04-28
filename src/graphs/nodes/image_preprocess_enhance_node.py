# -*- coding: utf-8 -*-
"""
图像预处理增强节点（V1.1新增）
实现文档方向分类、去畸变、去噪、图像增强等功能
"""

import os
import traceback
from typing import Dict, Any, List, Optional
from datetime import datetime
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context
from coze_coding_dev_sdk.s3 import S3SyncStorage

import cv2
import numpy as np
import requests

from utils.file.file import File, FileOps

from graphs.state import (
    ImagePreprocessEnhanceInput,
    ImagePreprocessEnhanceOutput
)


def _download_image(url: str) -> Optional[bytes]:
    """下载图片"""
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            return response.content
    except Exception as e:
        print(f"[图像预处理] 下载图片失败: {e}")
    return None


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
        print(f"[图像预处理] 上传图片失败: {e}")
        raise


def image_preprocess_enhance_node(state: ImagePreprocessEnhanceInput, config: RunnableConfig, runtime: Runtime[Context]) -> ImagePreprocessEnhanceOutput:
    """
    title: 图像预处理增强
    desc: 实现文档方向分类、去畸变、去噪、图像增强等功能，提升OCR识别准确率
    integrations: OpenCV, PaddleOCR
    """
    ctx = runtime.context

    print(f"[图像预处理增强] 开始处理图片...")
    print(f"[图像预处理增强] 配置: 方向分类={state.enable_orientation_classify}, 去畸变={state.enable_dewarp}, 去噪={state.enable_denoise}, 增强={state.enable_enhance}")

    try:
        start_time = datetime.now()
        enhancement_steps = []

        # 下载图片
        print(f"[图像预处理增强] 下载图片: {state.image.url}")
        img_data = _download_image(state.image.url)
        if img_data is None:
            raise Exception("图片下载失败")

        # 解码图片
        img_array = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if image is None:
            raise Exception("图片解码失败")

        print(f"[图像预处理增强] 原始图片尺寸: {image.shape}")
        processed_image = image.copy()
        orientation_angle = 0
        is_corrected = False

        # 1. 文档方向分类（使用PaddleOCR）
        if state.enable_orientation_classify:
            print(f"[图像预处理增强] 执行文档方向分类...")
            try:
                from paddleocr import PaddleOCR

                ocr = PaddleOCR(use_angle_cls=True, lang='ch', show_log=False)
                result = ocr.ocr(image, cls=True)

                if result and result[0]:
                    print(f"[图像预处理增强] OCR检测结果: {len(result[0])} 个文本块")
            except ImportError:
                print(f"[图像预处理增强] PaddleOCR未安装，跳过方向分类")
            except Exception as e:
                print(f"[图像预处理增强] 方向分类失败: {e}")

        # 2. 边缘投影法矫正小角度倾斜（±45度以内）
        if state.enable_dewarp:
            print(f"[图像预处理增强] 执行小角度矫正（边缘投影法）...")
            try:
                gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

                # 搜索最佳角度（默认±45度）
                best_angle = 0
                max_zero_count = 0
                angle_range = 45  # 固定使用45度范围

                for angle in range(-angle_range, angle_range + 1, 1):
                    (h, w) = binary.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    rotated = cv2.warpAffine(binary, M, (w, h))

                    horizontal_projection = np.sum(rotated, axis=1)
                    zero_count = np.sum(horizontal_projection == 0)

                    if zero_count > max_zero_count:
                        max_zero_count = zero_count
                        best_angle = angle

                if abs(best_angle) > 2:
                    print(f"[图像预处理增强] 检测到倾斜角度: {best_angle} 度")
                    (h, w) = processed_image.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
                    processed_image = cv2.warpAffine(processed_image, M, (w, h), flags=cv2.INTER_CUBIC)
                    orientation_angle = best_angle
                    is_corrected = True
                    enhancement_steps.append(f"倾斜矫正（{best_angle}度）")
            except Exception as e:
                print(f"[图像预处理增强] 倾斜矫正失败: {e}")

        # 3. 去噪（高斯模糊）
        if state.enable_denoise:
            print(f"[图像预处理增强] 执行去噪处理...")
            try:
                kernel_size = state.enhance_denoise_kernel
                denoised = cv2.GaussianBlur(processed_image, (kernel_size, kernel_size), 0)
                processed_image = denoised
                enhancement_steps.append(f"高斯去噪（kernel={kernel_size}）")
            except Exception as e:
                print(f"[图像预处理增强] 去噪失败: {e}")

        # 4. 图像增强
        if state.enable_enhance:
            print(f"[图像预处理增强] 执行图像增强...")
            try:
                lab = cv2.cvtColor(processed_image, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                l = clahe.apply(l)
                enhanced_lab = cv2.merge([l, a, b])
                enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
                enhanced_image = cv2.convertScaleAbs(enhanced_image, alpha=state.enhance_contrast, beta=0)

                kernel_sharpen = np.array([[-1, -1, -1],
                                           [-1, 9, -1],
                                           [-1, -1, -1]])
                sharpened = cv2.filter2D(enhanced_image, -1, kernel_sharpen)

                processed_image = sharpened
                enhancement_steps.append(f"对比度增强（{state.enhance_contrast}）")
                enhancement_steps.append("CLAHE增强")
                enhancement_steps.append("锐化处理")
            except Exception as e:
                print(f"[图像预处理增强] 图像增强失败: {e}")

        # 5. 上传处理后的图片
        print(f"[图像预处理增强] 上传处理后的图片...")
        file_name = f"preprocessed/enhanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        processed_image_url = _upload_image_to_storage(processed_image, file_name)

        processing_time = (datetime.now() - start_time).total_seconds()

        print(f"[图像预处理增强] 处理完成，耗时: {processing_time:.2f}秒")
        print(f"[图像预处理增强] 增强步骤: {enhancement_steps}")

        return ImagePreprocessEnhanceOutput(
            preprocessed_image=File(url=processed_image_url),
            orientation_angle=orientation_angle,
            is_corrected=is_corrected,
            enhancement_steps=enhancement_steps,
            processing_time=processing_time
        )

    except Exception as e:
        print(f"[图像预处理增强] 处理失败: {e}")
        traceback.print_exc()
        raise
