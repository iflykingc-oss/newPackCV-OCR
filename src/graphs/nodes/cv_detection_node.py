# -*- coding: utf-8 -*-
"""
CV目标检测节点
使用YOLOv8进行货架商品检测和定位
"""

import os
import traceback
import tempfile
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
    CVDetectionInput,
    CVDetectionOutput
)
from utils.file.file import File


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
        print(f"[CV检测] 上传图片失败: {e}")
        raise


def cv_detection_node(state: CVDetectionInput, config: RunnableConfig, runtime: Runtime[Context]) -> CVDetectionOutput:
    """
    title: CV目标检测
    desc: 使用YOLOv8检测货架上的商品，输出商品位置和置信度
    """
    ctx = runtime.context
    
    print(f"[CV检测] 开始处理货架图片...")
    
    try:
        # 下载图片
        print(f"[CV检测] 下载图片: {state.shelf_image.url}")
        img_data = download_image(state.shelf_image.url)
        if img_data is None:
            raise Exception("图片下载失败")
        
        # 解码图片
        img_array = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if image is None:
            raise Exception("图片解码失败")
        
        # 图像增强（如果启用）
        if state.enable_image_enhancement:
            image = enhance_image(image)
        
        # 使用YOLOv8进行目标检测
        start_time = datetime.now()
        detected_objects = detect_with_yolov8(image, state.detection_threshold)
        detection_time = (datetime.now() - start_time).total_seconds()
        
        # 统计结果
        total_count = len(detected_objects)
        avg_confidence = 0.0
        if detected_objects:
            avg_confidence = sum(obj.get('confidence', 0.0) for obj in detected_objects) / total_count
        
        # 生成标注图片（可选）
        processed_image = None
        if detected_objects:
            annotated_img = draw_detection_boxes(image.copy(), detected_objects)
            file_name = f"cv/detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            url = _upload_image_to_storage(annotated_img, file_name)
            processed_image = File(url=url)
        
        print(f"[CV检测] 检测完成，共 {total_count} 个商品，耗时 {detection_time:.2f} 秒")
        
        return CVDetectionOutput(
            detected_objects=detected_objects,
            total_count=total_count,
            detection_confidence=avg_confidence,
            processed_image=processed_image,
            detection_time=detection_time
        )
        
    except Exception as e:
        error_msg = f"CV检测节点发生错误: {str(e)}\n{traceback.format_exc()}"
        print(f"[CV检测] 错误: {error_msg}")
        
        return CVDetectionOutput(
            detected_objects=[],
            total_count=0,
            detection_confidence=0.0,
            processed_image=None,
            detection_time=0.0
        )


def download_image(image_url: str) -> Optional[bytes]:
    """下载图片"""
    try:
        response = requests.get(image_url, timeout=30)
        if response.status_code == 200:
            return response.content
        return None
    except Exception as e:
        print(f"下载图片失败: {str(e)}")
        return None


def enhance_image(image: np.ndarray) -> np.ndarray:
    """图像增强：去噪、增强对比度、校正"""
    try:
        # 去噪
        denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        
        # CLAHE增强对比度
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # 锐化
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        return sharpened
    except Exception as e:
        print(f"[CV检测] 图像增强失败，使用原图: {str(e)}")
        return image


def detect_with_yolov8(image: np.ndarray, threshold: float) -> List[Dict[str, Any]]:
    """使用YOLOv8进行目标检测"""
    try:
        from ultralytics import YOLO
        
        # 加载YOLOv8模型
        model = YOLO('yolov8n.pt')
        
        # 进行预测
        results = model(image, conf=threshold)
        
        # 解析结果
        detected_objects = []
        for result in results:
            boxes = result.boxes
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                class_name = result.names[class_id]
                
                detected_objects.append({
                    'id': i,
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(confidence),
                    'class_name': class_name,
                    'class_id': class_id,
                    'center': [(int(x1) + int(x2)) // 2, (int(y1) + int(y2)) // 2]
                })
        
        return detected_objects
        
    except ImportError:
        print("警告: ultralytics未安装，使用OpenCV轮廓检测作为降级方案")
        h, w = image.shape[:2]
        return _opencv_fallback_detection(image, h, w)
    except Exception as e:
        print(f"YOLOv8检测失败: {str(e)}，使用OpenCV轮廓检测作为降级方案")
        h, w = image.shape[:2]
        return _opencv_fallback_detection(image, h, w)


def _opencv_fallback_detection(image: np.ndarray, h: int, w: int) -> List[Dict[str, Any]]:
    """OpenCV降级方案：使用轮廓检测"""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_objects = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < 1000:
                continue
            x, y, bw, bh = cv2.boundingRect(contour)
            detected_objects.append({
                'id': i,
                'bbox': [x, y, x + bw, y + bh],
                'confidence': 0.75,
                'class_name': 'package',
                'class_id': 0,
                'center': [x + bw // 2, y + bh // 2]
            })
        return detected_objects
    except Exception as e:
        print(f"[CV检测] OpenCV轮廓检测失败: {e}")
        return []


def draw_detection_boxes(image: np.ndarray, detected_objects: List[Dict[str, Any]]) -> np.ndarray:
    """在图片上绘制检测框"""
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255)
    ]
    
    for obj in detected_objects:
        bbox = obj['bbox']
        x1, y1, x2, y2 = bbox
        confidence = obj['confidence']
        class_name = obj['class_name']
        
        color = colors[obj['id'] % len(colors)]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        label = f"{class_name} {confidence:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), color, -1)
        cv2.putText(image, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return image
