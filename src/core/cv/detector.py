# -*- coding: utf-8 -*-
"""
YOLO目标检测器
支持YOLOv8/YOLO11的标准检测和OBB有向边界框检测
"""

import os
import logging
from typing import List, Optional
import numpy as np

from core import (
    CVDetector, ROIObject, BoundingBox, OBBBox
)


logger = logging.getLogger(__name__)


class YOLODetector(CVDetector):
    """
    YOLO目标检测器
    支持：
    - 标准边界框检测 (BoundingBox)
    - 有向边界框检测 (OBB)
    """

    def __init__(
        self,
        model_path: str = "assets/models/yolo11n.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "cpu"
    ):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.model = None
        self._is_obb = "obb" in model_path.lower()

    def _load_model(self):
        """懒加载模型"""
        if self.model is None:
            try:
                from ultralytics import YOLO
                self.model = YOLO(self.model_path)
                logger.info(f"[YOLO检测] 模型加载成功: {self.model_path}")
            except ImportError:
                logger.error("[YOLO检测] ultralytics未安装，请执行: uv add ultralytics")
                raise ImportError("ultralytics未安装")

    def detect(self, image_path: str) -> List[ROIObject]:
        """
        标准边界框检测
        Returns: ROI对象列表
        """
        self._load_model()

        try:
            results = self.model.predict(
                source=image_path,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False
            )

            roi_objects = []
            for i, result in enumerate(results):
                boxes = result.boxes
                if boxes is None:
                    continue

                for box in boxes:
                    # 提取边界框坐标
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    label = result.names[cls_id]

                    roi = ROIObject(
                        roi_id=f"roi_{i}_{cls_id}",
                        bbox=BoundingBox(
                            x1=float(x1),
                            y1=float(y1),
                            x2=float(x2),
                            y2=float(y2),
                            score=conf
                        ),
                        confidence=conf,
                        class_id=str(cls_id),
                        label=label
                    )
                    roi_objects.append(roi)

            logger.info(f"[YOLO检测] 检测到 {len(roi_objects)} 个目标")
            return roi_objects

        except Exception as e:
            logger.error(f"[YOLO检测] 检测失败: {e}")
            raise

    def detect_obb(self, image_path: str) -> List[ROIObject]:
        """
        有向边界框(OBB)检测
        适用于倾斜/堆叠商品的精准定位
        Returns: ROI对象列表
        """
        self._load_model()

        try:
            # OBB模式检测
            results = self.model.predict(
                source=image_path,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False,
                task='obb' if self._is_obb else 'detect'
            )

            roi_objects = []
            for i, result in enumerate(results):
                # 检查是否支持OBB
                if not hasattr(result, 'obb') or result.obb is None:
                    # 如果不支持OBB，回退到标准检测
                    logger.warning("[YOLO检测] 模型不支持OBB，回退到标准检测")
                    return self.detect(image_path)

                obb = result.obb
                for j, box in enumerate(obb):
                    # 提取OBB角点
                    xy = box.xyxyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    label = result.names[cls_id]

                    # 转换为4角点列表
                    corners = [[float(p[0]), float(p[1])] for p in xy]

                    roi = ROIObject(
                        roi_id=f"obb_roi_{i}_{j}",
                        bbox=OBBBox(
                            corners=corners,
                            score=conf,
                            class_id=str(cls_id),
                            label=label
                        ),
                        confidence=conf,
                        class_id=str(cls_id),
                        label=label
                    )
                    roi_objects.append(roi)

            logger.info(f"[YOLO检测] OBB检测到 {len(roi_objects)} 个目标")
            return roi_objects

        except Exception as e:
            logger.error(f"[YOLO检测] OBB检测失败: {e}")
            # 降级到标准检测
            return self.detect(image_path)


def create_detector(
    model_path: Optional[str] = None,
    conf_threshold: float = 0.25,
    device: str = "cpu"
) -> YOLODetector:
    """
    创建YOLO检测器工厂函数
    """
    if model_path is None:
        # 默认使用YOLO11n模型
        model_path = "assets/models/yolo11n.pt"

    return YOLODetector(
        model_path=model_path,
        conf_threshold=conf_threshold,
        device=device
    )
