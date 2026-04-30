# -*- coding: utf-8 -*-
"""
图像预处理模块
提供标准化的图像预处理算子
"""

import cv2
import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass

from core import ImageFormat


logger = logging.getLogger(__name__)


@dataclass
class PreprocessConfig:
    """预处理配置"""
    denoise: bool = True
    denoise_strength: int = 10
    enhance_contrast: bool = True
    clahe_clip_limit: float = 2.0
    clahe_grid_size: Tuple[int, int] = (8, 8)
    sharpen: bool = True
    sharpen_amount: float = 1.5
    perspective_correct: bool = False
    remove_glare: bool = False
    low_light_enhance: bool = False


class ImagePreprocessor:
    """
    标准化图像预处理算子
    支持：
    - 去噪（高斯滤波）
    - 对比度增强（CLAHE）
    - 锐化（unsharp mask）
    - 透视矫正
    - 反光去除
    - 低光照增强
    """

    def __init__(self, config: Optional[PreprocessConfig] = None):
        self.config = config or PreprocessConfig()

    def preprocess(self, image_path: str) -> Dict[str, Any]:
        """
        执行完整预处理流程
        Returns: {
            'processed_image': bytes,
            'image_format': ImageFormat,
            'metadata': {...}
        }
        """
        # 读取图片
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图片: {image_path}")

        original_shape = img.shape
        metadata = {
            'original_shape': original_shape,
            'preprocessing_steps': []
        }

        # 1. 去噪
        if self.config.denoise:
            img = self.denoise(img)
            metadata['preprocessing_steps'].append('denoise')

        # 2. 低光照增强
        if self.config.low_light_enhance:
            img = self.enhance_low_light(img)
            metadata['preprocessing_steps'].append('low_light_enhance')

        # 3. 对比度增强
        if self.config.enhance_contrast:
            img = self.enhance_contrast_clahe(img)
            metadata['preprocessing_steps'].append('contrast_enhance')

        # 4. 锐化
        if self.config.sharpen:
            img = self.sharpen(img)
            metadata['preprocessing_steps'].append('sharpen')

        # 5. 反光去除
        if self.config.remove_glare:
            img = self.remove_glare(img)
            metadata['preprocessing_steps'].append('glare_removal')

        # 6. 透视矫正
        if self.config.perspective_correct:
            img, transform_matrix = self.correct_perspective(img)
            metadata['preprocessing_steps'].append('perspective_correct')
            metadata['transform_matrix'] = transform_matrix

        # 编码为bytes
        _, buffer = cv2.imencode('.png', img)
        processed_bytes = buffer.tobytes()

        logger.info(f"[预处理] 完成 {len(metadata['preprocessing_steps'])} 步处理")

        return {
            'processed_image': processed_bytes,
            'image_format': ImageFormat.PNG,
            'metadata': metadata
        }

    def denoise(self, image: np.ndarray) -> np.ndarray:
        """高斯去噪"""
        return cv2.fastNlMeansDenoisingColored(
            image, None,
            self.config.denoise_strength,
            self.config.denoise_strength,
            7, 21
        )

    def enhance_contrast_clahe(self, image: np.ndarray) -> np.ndarray:
        """CLAHE对比度增强"""
        # 转为LAB色彩空间
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # 应用CLAHE
        clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=self.config.clahe_grid_size
        )
        l = clahe.apply(l)

        # 合并
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def sharpen(self, image: np.ndarray) -> np.ndarray:
        """锐化（Unsharp Mask）"""
        gaussian = cv2.GaussianBlur(image, (0, 0), self.config.sharpen_amount)
        sharpened = cv2.addWeighted(image, 1.0 + self.config.sharpen_amount, gaussian, -self.config.sharpen_amount, 0)
        return sharpened

    def enhance_low_light(self, image: np.ndarray) -> np.ndarray:
        """低光照增强（MSRCR算法简化版）"""
        # 转为浮点类型
        img_float = np.float64(image) + 1.0

        # 对数变换
        log_img = np.log(img_float)

        # 多尺度Retinex
        scales = [15, 80, 200]
        msr = np.zeros_like(log_img)

        for sigma in scales:
            blur = cv2.GaussianBlur(img_float, (0, 0), sigma)
            msr += np.log(img_float / blur + 1e-10)

        msr = msr / len(scales)

        # 归一化
        msr = (msr - msr.min()) / (msr.max() - msr.min() + 1e-10) * 255
        return np.clip(msr, 0, 255).astype(np.uint8)

    def remove_glare(self, image: np.ndarray) -> np.ndarray:
        """反光去除（基于亮度阈值）"""
        # 转为HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 分离通道
        h, s, v = cv2.split(hsv)

        # 高亮度区域（反光）阈值
        _, glare_mask = cv2.threshold(v, 220, 255, cv2.THRESH_BINARY)

        # 用邻近非反光区域填充
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(v, kernel, iterations=2)

        # 平滑处理
        inpaint = cv2.inpaint(image, glare_mask, 3, cv2.INPAINT_TELEA)

        return inpaint

    def correct_perspective(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        透视矫正
        需要配合边缘检测使用，这里提供简化版本
        """
        # 获取图像尺寸
        height, width = image.shape[:2]

        # 默认透视变换矩阵（无变换）
        transform_matrix = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ], dtype=np.float32)

        # 实际使用中需要检测四边形的四个角点
        # 这里返回原始图像
        return image, transform_matrix


def create_preprocessor(
    denoise: bool = True,
    enhance_contrast: bool = True,
    sharpen: bool = True,
    low_light_enhance: bool = False,
    remove_glare: bool = False
) -> ImagePreprocessor:
    """预处理工厂函数"""
    config = PreprocessConfig(
        denoise=denoise,
        enhance_contrast=enhance_contrast,
        sharpen=sharpen,
        low_light_enhance=low_light_enhance,
        remove_glare=remove_glare
    )
    return ImagePreprocessor(config)
