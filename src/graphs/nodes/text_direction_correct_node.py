# -*- coding: utf-8 -*-
"""
文本方向矫正节点（V1.1新增）
使用边缘投影法和PaddleOCR分类模型进行文本方向矫正
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
    TextDirectionCorrectInput,
    TextDirectionCorrectOutput
)


def _download_image(url: str) -> Optional[bytes]:
    """下载图片"""
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            return response.content
    except Exception as e:
        print(f"[文本方向矫正] 下载图片失败: {e}")
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
        print(f"[文本方向矫正] 上传图片失败: {e}")
        raise


def text_direction_correct_node(state: TextDirectionCorrectInput, config: RunnableConfig, runtime: Runtime[Context]) -> TextDirectionCorrectOutput:
    """
    title: 文本方向矫正
    desc: 使用边缘投影法和PaddleOCR分类模型进行文本方向矫正，支持0-360度旋转文本识别
    integrations: OpenCV, PaddleOCR
    """
    ctx = runtime.context

    print(f"[文本方向矫正] 开始处理图片...")
    print(f"[文本方向矫正] 配置: 边缘投影法={state.use_edge_projection}, 分类模型={state.use_cls_model}, 角度范围=+-{state.angle_range}度")

    try:
        start_time = datetime.now()

        # 下载图片
        print(f"[文本方向矫正] 下载图片: {state.image.url}")
        img_data = _download_image(state.image.url)
        if img_data is None:
            raise Exception("图片下载失败")

        # 解码图片
        img_array = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if image is None:
            raise Exception("图片解码失败")

        print(f"[文本方向矫正] 原始图片尺寸: {image.shape}")

        detected_angle = 0.0
        correction_method = ""
        confidence = 0.0
        corrected_image = image.copy()

        # 方法1：边缘投影法（适用于小角度倾斜 ±45度）
        if state.use_edge_projection:
            print(f"[文本方向矫正] 使用边缘投影法检测倾斜角度...")
            try:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

                best_angle = 0
                max_zero_count = 0
                angles = range(-state.angle_range, state.angle_range + 1, 1)

                for angle in angles:
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
                    print(f"[文本方向矫正] 边缘投影法检测到倾斜角度: {best_angle} 度")
                    (h, w) = image.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
                    corrected_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

                    detected_angle = float(best_angle)
                    correction_method = "edge_projection"
                    confidence = min(max_zero_count / (h * w), 1.0)
                else:
                    print(f"[文本方向矫正] 检测角度过小（{best_angle}度），无需矫正")
            except Exception as e:
                print(f"[文本方向矫正] 边缘投影法失败: {e}")

        # 方法2：PaddleOCR角度分类（适用于大角度旋转 90/180/270度）
        if state.use_cls_model and detected_angle == 0:
            print(f"[文本方向矫正] 使用PaddleOCR分类模型检测方向...")
            try:
                from paddleocr import PaddleOCR

                ocr = PaddleOCR(use_angle_cls=True, lang='ch', show_log=False)
                result = ocr.ocr(image, cls=True)

                if result and result[0]:
                    print(f"[文本方向矫正] OCR检测到 {len(result[0])} 个文本块")

                    boxes = [item[0] for item in result[0] if len(item) > 0]

                    if boxes:
                        angle_list = []
                        for box in boxes:
                            if len(box) >= 2:
                                p1 = box[0]
                                p2 = box[1]
                                dx = p2[0] - p1[0]
                                dy = p2[1] - p1[1]
                                angle = np.arctan2(dy, dx) * 180 / np.pi
                                angle_list.append(angle)

                        if angle_list:
                            avg_angle = np.mean(angle_list)

                            if abs(avg_angle) > 45:
                                rotation = 90 * round(avg_angle / 90)
                                if rotation != 0:
                                    print(f"[文本方向矫正] 检测到大角度旋转: {rotation} 度")
                                    (h, w) = image.shape[:2]
                                    center = (w // 2, h // 2)
                                    M = cv2.getRotationMatrix2D(center, rotation, 1.0)
                                    corrected_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

                                    detected_angle = float(rotation)
                                    correction_method = "paddleocr_cls"
                                    confidence = 0.85
            except ImportError:
                print(f"[文本方向矫正] PaddleOCR未安装，跳过角度分类")
            except Exception as e:
                print(f"[文本方向矫正] PaddleOCR角度分类失败: {e}")

        # 上传矫正后的图片
        print(f"[文本方向矫正] 上传矫正后的图片...")
        file_name = f"corrected/corrected_{datetime.now().strftime('%Y%m%d_%H%M%S')}_angle{detected_angle:.1f}.jpg"
        corrected_image_url = _upload_image_to_storage(corrected_image, file_name)

        processing_time = (datetime.now() - start_time).total_seconds()

        print(f"[文本方向矫正] 处理完成，耗时: {processing_time:.2f}秒")
        print(f"[文本方向矫正] 检测角度: {detected_angle}度, 方法: {correction_method}, 置信度: {confidence:.2f}")

        return TextDirectionCorrectOutput(
            corrected_image=File(url=corrected_image_url),
            detected_angle=detected_angle,
            correction_method=correction_method,
            confidence=confidence,
            processing_time=processing_time
        )

    except Exception as e:
        print(f"[文本方向矫正] 处理失败: {e}")
        traceback.print_exc()
        raise
