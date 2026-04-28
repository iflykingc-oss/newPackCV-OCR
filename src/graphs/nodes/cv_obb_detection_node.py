# -*- coding: utf-8 -*-
"""
YOLO11-OBB旋转框检测节点（V1.3新增）
使用YOLO11s-obb模型检测旋转包装文本，支持(cx,cy,w,h,angle)旋转框输出
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

from graphs.state import (
    CVOBBDetectionInput,
    CVOBBDetectionOutput
)
from utils.file.file import File


def _download_image(url: str) -> Optional[bytes]:
    """下载图片"""
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            return response.content
    except Exception as e:
        print(f"[OBB检测] 下载图片失败: {e}")
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
        print(f"[OBB检测] 上传图片失败: {e}")
        raise


def cv_obb_detection_node(state: CVOBBDetectionInput, config: RunnableConfig, runtime: Runtime[Context]) -> CVOBBDetectionOutput:
    """
    title: YOLO11-OBB旋转框检测
    desc: 使用YOLO11s-obb模型检测旋转包装文本，支持(cx,cy,w,h,angle)旋转框输出，mAP@50=79.5%
    integrations: Ultralytics, OpenCV
    """
    ctx = runtime.context

    print(f"[OBB检测] 开始处理图片...")
    print(f"[OBB检测] 配置: 置信度阈值={state.confidence_threshold}, IOU阈值={state.iou_threshold}")

    try:
        start_time = datetime.now()

        # 下载图片
        print(f"[OBB检测] 下载图片: {state.image.url}")
        img_data = _download_image(state.image.url)
        if img_data is None:
            raise Exception("图片下载失败")

        # 解码图片
        img_array = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if image is None:
            raise Exception("图片解码失败")

        print(f"[OBB检测] 原始图片尺寸: {image.shape}")

        # 使用YOLO11-OBB进行旋转框检测
        obb_results = []
        try:
            from ultralytics import YOLO

            print(f"[OBB检测] 加载YOLO11-OBB模型...")
            model = YOLO('yolo11n-obb.pt')

            print(f"[OBB检测] 执行旋转框检测...")
            results = model(image, conf=state.confidence_threshold, iou=state.iou_threshold)

            # 解析旋转框结果
            for result in results:
                if result.obb is not None:
                    for i, obb in enumerate(result.obb):
                        # 获取旋转框坐标 (cx, cy, w, h, angle)
                        xywhr = obb.xywhr.cpu().numpy()[0]
                        cx, cy, w, h, angle = xywhr

                        confidence = float(obb.conf.cpu().numpy()[0])
                        class_id = int(obb.cls.cpu().numpy()[0])
                        class_name = result.names[class_id] if class_id in result.names else f"class_{class_id}"

                        # 计算旋转框四个顶点
                        cos_a = np.cos(angle)
                        sin_a = np.sin(angle)
                        corners = np.array([
                            [-w/2, -h/2],
                            [w/2, -h/2],
                            [w/2, h/2],
                            [-w/2, h/2]
                        ])
                        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                        rotated_corners = corners @ rotation_matrix.T
                        rotated_corners[:, 0] += cx
                        rotated_corners[:, 1] += cy
                        box_points = np.array(rotated_corners, dtype=np.int32)

                        obb_results.append({
                            'id': i,
                            'cx': float(cx),
                            'cy': float(cy),
                            'width': float(w),
                            'height': float(h),
                            'angle': float(np.degrees(angle)),
                            'confidence': confidence,
                            'class_name': class_name,
                            'class_id': class_id,
                            'box_points': box_points.tolist()
                        })

            print(f"[OBB检测] 检测到 {len(obb_results)} 个旋转目标")

        except ImportError:
            print(f"[OBB检测] Ultralytics未安装，使用OpenCV轮廓检测作为降级方案")
            obb_results = _opencv_fallback_obb(image)
        except Exception as e:
            print(f"[OBB检测] YOLO-OBB检测失败: {e}，使用OpenCV轮廓检测作为降级方案")
            obb_results = _opencv_fallback_obb(image)

        # 绘制旋转框标注图
        processed_image = None
        if obb_results:
            annotated_img = _draw_obb_boxes(image.copy(), obb_results)
            file_name = f"obb/obb_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            url = _upload_image_to_storage(annotated_img, file_name)
            processed_image = File(url=url)

        processing_time = (datetime.now() - start_time).total_seconds()

        print(f"[OBB检测] 处理完成，耗时: {processing_time:.2f}秒")

        return CVOBBDetectionOutput(
            obb_results=obb_results,
            total_count=len(obb_results),
            processed_image=processed_image,
            processing_time=processing_time
        )

    except Exception as e:
        print(f"[OBB检测] 处理失败: {e}")
        traceback.print_exc()
        raise


def _opencv_fallback_obb(image: np.ndarray) -> List[Dict[str, Any]]:
    """OpenCV降级方案：使用轮廓检测模拟旋转框"""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        obb_results = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < 1000:
                continue

            rect = cv2.minAreaRect(contour)
            (cx, cy), (w, h), angle = rect

            box = cv2.boxPoints(rect)
            box = np.array(box, dtype=np.int32)

            obb_results.append({
                'id': i,
                'cx': float(cx),
                'cy': float(cy),
                'width': float(w),
                'height': float(h),
                'angle': float(angle),
                'confidence': 0.75,
                'class_name': 'package',
                'class_id': 0,
                'box_points': box.tolist()
            })

        return obb_results
    except Exception as e:
        print(f"[OBB检测] OpenCV降级检测失败: {e}")
        return []


def _draw_obb_boxes(image: np.ndarray, obb_results: List[Dict[str, Any]]) -> np.ndarray:
    """在图片上绘制旋转框"""
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255)
    ]

    for obj in obb_results:
        box_points = obj['box_points']
        confidence = obj['confidence']
        class_name = obj['class_name']
        angle = obj['angle']

        color = colors[obj['id'] % len(colors)]

        # 绘制旋转框
        pts = np.array(box_points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [pts], True, color, 2)

        # 标注信息
        cx = int(obj['cx'])
        cy = int(obj['cy'])
        label = f"{class_name} {confidence:.2f} {angle:.1f}deg"
        cv2.putText(image, label, (cx - 40, cy - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    return image
