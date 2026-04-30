# -*- coding: utf-8 -*-
"""
CV视觉层模块
提供图像预处理、目标检测、ROI裁切能力
"""

from core.cv.detector import YOLODetector, CVDetector
from core.cv.preprocessor import ImagePreprocessor, CVPreprocessor
from core.cv.cropper import ROICropper, ImageCropper
from core import BoundingBox, OBBBox, ROIObject

__all__ = [
    'YOLODetector',
    'CVDetector',
    'ImagePreprocessor',
    'CVPreprocessor',
    'ROICropper',
    'ImageCropper',
    'BoundingBox',
    'OBBBox',
    'ROIObject'
]
