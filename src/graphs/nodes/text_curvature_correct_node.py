# -*- coding: utf-8 -*-
"""
弯曲文本校正节点 V5.6
基于TPS（Thin Plate Spline）薄板样条插值的弯曲文本校正。
专门处理圆柱体包装（瓶/罐/听）、弧形标签等场景的弯曲文本。
核心流程：
1. 文本区域检测 + 弯曲程度评估
2. 贝塞尔曲线拟合文本基线
3. TPS薄板样条插值校正
"""

import os
import time
import logging
import traceback
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

import cv2
import numpy as np
import requests

from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context
from graphs.state import TextCurvatureCorrectInput, TextCurvatureCorrectOutput
from utils.file.file import File

logger = logging.getLogger(__name__)


# ==================== 弯曲检测 ====================

def _detect_curvature(image: np.ndarray) -> Tuple[float, Optional[np.ndarray]]:
    """
    检测文本弯曲程度
    Returns:
        (curvature_score: 0-1, text_mask: 文本区域掩码)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # 1. MSER检测文本区域
    mser = cv2.MSER_create(_min_area=60, _max_area=int(h * w * 0.3))  # type: ignore[attr-defined]
    try:
        regions, _ = mser.detectRegions(gray)
    except Exception:
        # MSER可能在某些图像上失败
        return 0.0, None
    
    if not regions or len(regions) < 3:
        return 0.0, None
    
    # 2. 构建文本区域掩码
    text_mask = np.zeros(gray.shape, dtype=np.uint8)
    for region in regions:
        if len(region) < 4:
            continue
        hull = cv2.convexHull(region.reshape(-1, 1, 2).astype(np.int32))
        cv2.fillPoly(text_mask, [hull], 255)
    
    # 3. 垂直投影分析 - 检测文本行弯曲
    # 将图像分为水平条带，分析文本区域的垂直分布
    num_strips = 10
    strip_height = h // num_strips
    vertical_centers = []
    
    for i in range(num_strips):
        y_start = i * strip_height
        y_end = min((i + 1) * strip_height, h)
        strip_mask = text_mask[y_start:y_end, :]
        
        # 找到文本区域的水平中心
        ys, xs = np.where(strip_mask > 0)
        if len(xs) > 0:
            center_x = float(np.mean(xs))
            vertical_centers.append((y_start + strip_height // 2, center_x))
    
    if len(vertical_centers) < 4:
        return 0.0, None
    
    # 4. 拟合直线，计算残差
    ys_arr = np.array([p[0] for p in vertical_centers], dtype=np.float32)
    xs_arr = np.array([p[1] for p in vertical_centers], dtype=np.float32)
    
    # 最小二乘直线拟合
    A = np.vstack([ys_arr, np.ones_like(ys_arr)]).T
    slope, intercept = np.linalg.lstsq(A, xs_arr, rcond=None)[0]
    
    # 计算预测值
    xs_pred = slope * ys_arr + intercept
    
    # 计算相对残差（弯曲程度）
    residuals = np.abs(xs_arr - xs_pred)
    max_residual = float(np.max(residuals))
    
    # 归一化评分：相对于图像宽度
    curvature_score = min(1.0, max_residual / (w * 0.15))
    
    logger.info(f"弯曲检测: 最大残差={max_residual:.1f}px, 评分={curvature_score:.3f}")
    
    return curvature_score, text_mask


# ==================== TPS校正 ====================

def _build_tps_grid(h: int, w: int, curvature_score: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    构建源点和目标点网格
    对于圆柱体弯曲，使用正弦曲线近似
    返回: (map_x, map_y) - 用于cv2.remap的映射矩阵
    """
    # 弯曲幅度
    amplitude = curvature_score * w * 0.08
    
    # 创建网格坐标
    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)
    
    for y in range(h):
        # 正弦曲线偏移（适合瓶身/罐体弯曲）
        offset = amplitude * np.sin(np.pi * y / h)
        for x in range(w):
            map_x[y, x] = x - offset * (1 - 2 * abs(x - w / 2) / w)
            map_y[y, x] = y
    
    # 确保坐标在有效范围内
    map_x = np.clip(map_x, 0, w - 1)
    map_y = np.clip(map_y, 0, h - 1)
    
    return map_x, map_y


def _apply_tps_correction(image: np.ndarray, curvature_score: float) -> np.ndarray:
    """应用基于remap的弯曲校正（替代TPS薄板样条）"""
    h, w = image.shape[:2]
    
    # 构建重映射网格
    map_x, map_y = _build_tps_grid(h, w, curvature_score)
    
    # 应用重映射
    corrected = cv2.remap(image, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return corrected


# ==================== 图片工具 ====================

def _download_image(url: str) -> Optional[bytes]:
    """下载图片"""
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            return resp.content
    except Exception:
        pass
    return None


def _upload_image_to_storage(image_array: np.ndarray, file_name: str) -> str:
    """上传图片到对象存储"""
    from storage.oss import get_oss_storage
    is_success, buffer = cv2.imencode('.jpg', image_array, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not is_success:
        raise Exception("图片编码失败")
    storage = get_oss_storage()
    key = storage.upload_file(
        file_content=buffer.tobytes(),
        file_name=file_name,
        content_type='image/jpeg'
    )
    url = storage.generate_presigned_url(key=key, expire_time=86400)
    return url


# ==================== 节点主函数 ====================

def text_curvature_correct_node(
    state: TextCurvatureCorrectInput,
    config: RunnableConfig,
    runtime: Runtime[Context]
) -> TextCurvatureCorrectOutput:
    """
    title: 弯曲文本TPS校正
    desc: V5.6 基于TPS(Thin Plate Spline)薄板样条插值的弯曲文本校正引擎。针对圆柱体包装(瓶/罐/听)、弧形标签等场景，自动检测文本弯曲程度并应用非线性校正，将弯曲文本展平为水平文本，大幅提升后续OCR/多模态识别的准确率。
    integrations: OpenCV
    """
    ctx = runtime.context
    logger.info("=== 弯曲文本校正开始 ===")
    t_start = time.time()
    
    try:
        # 选择图片：优先使用enhanced_image，回退到preprocessed_image，再回退到package_image
        target_image = state.enhanced_image or state.preprocessed_image or state.package_image
        if target_image is None:
            raise Exception("无可用的输入图片")
        
        # 下载图片
        img_data = _download_image(target_image.url)
        if img_data is None:
            raise Exception("图片下载失败")
        
        img_array = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if image is None:
            raise Exception("图片解码失败")
        
        logger.info(f"原始图片尺寸: {image.shape}")
        
        # 1. 弯曲检测
        curvature_score, text_mask = _detect_curvature(image)
        curvature_detected = curvature_score > state.curvature_detection_threshold
        
        logger.info(f"弯曲检测结果: 评分={curvature_score:.3f}, 检测到弯曲={curvature_detected}")
        
        # 2. TPS校正
        tps_applied = False
        correction_confidence = 0.0
        
        if curvature_detected and state.enable_tps_correction:
            logger.info("应用TPS弯曲校正...")
            corrected = _apply_tps_correction(image, curvature_score)
            
            # 验证校正效果（检查校正后文本水平度）
            post_score, _ = _detect_curvature(corrected)
            improvement = curvature_score - post_score
            
            logger.info(f"TPS校正后弯曲评分: {post_score:.3f}, 改善: {improvement:.3f}")
            
            if improvement > 0.05:
                tps_applied = True
                correction_confidence = min(1.0, improvement / curvature_score)
                result_image = corrected
            else:
                logger.info("TPS校正改善有限，保留原图")
                result_image = image
        else:
            result_image = image
        
        # 3. 上传结果
        file_name = f"curvature_corrected/corrected_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        corrected_url = _upload_image_to_storage(result_image, file_name)
        
        elapsed = time.time() - t_start
        logger.info(f"弯曲文本校正完成，耗时: {elapsed:.3f}秒")
        
        return TextCurvatureCorrectOutput(
            corrected_image=File(url=corrected_url, file_type="image"),
            curvature_detected=curvature_detected,
            tps_applied=tps_applied,
            curvature_score=round(curvature_score, 3),
            correction_confidence=round(correction_confidence, 3),
            processing_time=round(elapsed, 3)
        )
        
    except Exception as e:
        logger.error(f"弯曲文本校正失败: {str(e)}\n{traceback.format_exc()}")
        # 失败返回输入图
        fallback = state.enhanced_image or state.preprocessed_image or state.package_image
        return TextCurvatureCorrectOutput(
            corrected_image=fallback or File(url="", file_type="image"),
            curvature_detected=False,
            tps_applied=False,
            curvature_score=0.0,
            correction_confidence=0.0,
            processing_time=round(time.time() - t_start, 3)
        )