# -*- coding: utf-8 -*-
"""
ROI裁切模块
提供标准化ROI裁切和NMS去重能力
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple

from core import ROICropper, ROIObject, BoundingBox, OBBBox


logger = logging.getLogger(__name__)


class ImageCropper(ROICropper):
    """
    ROI裁切器
    支持：
    - 标准边界框裁切
    - OBB有向边界框裁切
    - NMS非极大值抑制
    - 边缘补全
    """

    def __init__(
        self,
        padding: int = 5,
        min_crop_size: int = 20,
        target_size: Tuple[int, int] = None
    ):
        """
        Args:
            padding: 裁切边缘padding像素
            min_crop_size: 最小裁切尺寸
            target_size: 统一输出尺寸（可选）
        """
        self.padding = padding
        self.min_crop_size = min_crop_size
        self.target_size = target_size

    def crop(self, image_path: str, roi_objects: List[ROIObject]) -> List[ROIObject]:
        """
        执行ROI裁切
        Returns: 包含裁切图的ROI对象列表
        """
        # 读取原图
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图片: {image_path}")

        img_height, img_width = img.shape[:2]
        cropped_objects = []

        for roi in roi_objects:
            try:
                if isinstance(roi.bbox, OBBBox):
                    # OBB裁切
                    cropped = self._crop_obb(img, roi.bbox, img_width, img_height)
                else:
                    # 标准边界框裁切
                    cropped = self._crop_bbox(img, roi.bbox, img_width, img_height)

                if cropped is None:
                    continue

                # 统一尺寸
                if self.target_size:
                    cropped = cv2.resize(cropped, self.target_size)

                # 编码为bytes
                _, buffer = cv2.imencode('.png', cropped)
                roi.crop_image = buffer.tobytes()

                cropped_objects.append(roi)
                logger.debug(f"[裁切] ROI {roi.roi_id} 裁切成功")

            except Exception as e:
                logger.warning(f"[裁切] ROI {roi.roi_id} 裁切失败: {e}")
                continue

        logger.info(f"[裁切] 成功裁切 {len(cropped_objects)}/{len(roi_objects)} 个ROI")
        return cropped_objects

    def _crop_bbox(
        self,
        img: np.ndarray,
        bbox: BoundingBox,
        img_width: int,
        img_height: int
    ) -> np.ndarray:
        """裁切标准边界框区域"""
        # 添加padding
        x1 = max(0, int(bbox.x1) - self.padding)
        y1 = max(0, int(bbox.y1) - self.padding)
        x2 = min(img_width, int(bbox.x2) + self.padding)
        y2 = min(img_height, int(bbox.y2) + self.padding)

        # 检查尺寸
        if x2 - x1 < self.min_crop_size or y2 - y1 < self.min_crop_size:
            return None

        # 裁切
        return img[y1:y2, x1:x2]

    def _crop_obb(
        self,
        img: np.ndarray,
        obb: OBBBox,
        img_width: int,
        img_height: int
    ) -> np.ndarray:
        """
        裁切OBB区域
        使用透视变换将倾斜区域转为矩形
        """
        corners = np.array(obb.corners, dtype=np.float32)

        # 计算宽高
        width = int(max(
            np.linalg.norm(corners[0] - corners[1]),
            np.linalg.norm(corners[2] - corners[3])
        ))
        height = int(max(
            np.linalg.norm(corners[0] - corners[3]),
            np.linalg.norm(corners[1] - corners[2])
        ))

        # 目标矩形角点
        dst_corners = np.array([
            [0, 0],
            [width, 0],
            [width, height],
            [0, height]
        ], dtype=np.float32)

        # 计算透视变换矩阵
        M = cv2.getPerspectiveTransform(corners, dst_corners)

        # 应用透视变换
        warped = cv2.warpPerspective(img, M, (width, height))

        # 边缘补全（处理透视变换后的黑边）
        warped = self._fill_edges(warped)

        return warped

    def _fill_edges(self, img: np.ndarray) -> np.ndarray:
        """边缘补全"""
        # 检测非黑色边缘区域
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

        # 膨胀mask
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)

        # 使用膨胀后的内容填充边缘
        img_filled = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
        return img_filled

    def apply_nms(
        self,
        roi_objects: List[ROIObject],
        iou_threshold: float = 0.5
    ) -> List[ROIObject]:
        """
        非极大值抑制
        去除重叠的ROI，保留最高置信度的
        """
        if not roi_objects:
            return []

        # 提取边界框和置信度
        boxes = []
        scores = []
        indices = []

        for i, roi in enumerate(roi_objects):
            if isinstance(roi.bbox, BoundingBox):
                boxes.append([
                    roi.bbox.x1, roi.bbox.y1,
                    roi.bbox.x2, roi.bbox.y2
                ])
                scores.append(roi.confidence)
                indices.append(i)

        if not boxes:
            return roi_objects

        # 转为numpy数组
        boxes = np.array(boxes)
        scores = np.array(scores)

        # 执行NMS
        keep_indices = self._nms(boxes, scores, iou_threshold)

        # 返回保留的ROI
        result = [roi_objects[indices[i]] for i in keep_indices]
        removed = len(roi_objects) - len(result)
        if removed > 0:
            logger.info(f"[NMS] 移除了 {removed} 个重叠ROI")

        return result

    def _nms(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        iou_threshold: float
    ) -> List[int]:
        """
        非极大值抑制实现
        """
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        # 计算面积
        areas = (x2 - x1) * (y2 - y1)

        # 按置信度排序
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            # 计算与其他框的IOU
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter)

            # 保留IOU低于阈值的框
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return keep


def create_cropper(
    padding: int = 5,
    target_size: Tuple[int, int] = (640, 640)
) -> ImageCropper:
    """裁切器工厂函数"""
    return ImageCropper(padding=padding, target_size=target_size)
