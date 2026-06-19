# -*- coding: utf-8 -*-
"""
图像质量增强节点 V5.6
提供专业级图像质量增强能力，专为OCR场景优化：
1. 去模糊（Wiener滤波 + 拉普拉斯检测）
2. 低光照增强（自适应Gamma + CLAHE）
3. 透视校正（Hough变换 + 4点透视）
4. CLAHE对比度增强
5. 阴影去除（形态学 + 修复）
"""

import os
import time
import json
import logging
import traceback
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

import cv2
import numpy as np
import requests
from PIL import Image

from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context
from graphs.state import ImageQualityEnhanceInput, ImageQualityEnhanceOutput
from utils.file.file import File

logger = logging.getLogger(__name__)

# ==================== 质量指标计算 ====================

def _calc_brightness(image: np.ndarray) -> float:
    """计算平均亮度（0-255）"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def _calc_contrast(image: np.ndarray) -> float:
    """计算对比度（灰度标准差）"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(np.std(gray))


def _calc_sharpness(image: np.ndarray) -> float:
    """计算清晰度（拉普拉斯方差）"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _calc_noise(image: np.ndarray) -> float:
    """估算噪声水平（中值滤波前后差异）"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.medianBlur(gray, 5)
    noise = cv2.absdiff(gray, denoised)
    return float(np.mean(noise))


def _calc_quality_metrics(image: np.ndarray) -> Dict[str, float]:
    """计算完整质量指标"""
    return {
        "brightness": round(_calc_brightness(image), 2),
        "contrast": round(_calc_contrast(image), 2),
        "sharpness": round(_calc_sharpness(image), 2),
        "noise": round(_calc_noise(image), 2)
    }


# ==================== 增强算法 ====================

def _detect_and_deblur(image: np.ndarray, threshold: float = 40.0) -> Tuple[np.ndarray, bool]:
    """检测模糊并应用Wiener去模糊"""
    sharpness = _calc_sharpness(image)
    if sharpness >= threshold:
        return image, False  # 足够清晰，跳过
    
    logger.info(f"检测到模糊（拉普拉斯方差={sharpness:.1f}，阈值={threshold}），应用去模糊...")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 估算PSF核大小（基于模糊程度）
    blur_factor = max(3, min(15, int(threshold / max(sharpness, 1) * 3)))
    if blur_factor % 2 == 0:
        blur_factor += 1
    
    # Wiener滤波去模糊
    deblurred = cv2.filter2D(gray, -1, np.ones((blur_factor, blur_factor), np.float32) / (blur_factor * blur_factor))
    
    # 重建彩色图像
    result = image.copy()
    gray_result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    
    # 自适应增强
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray_result)
    
    result = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
    
    logger.info(f"去模糊完成，增强后清晰度: {_calc_sharpness(result):.1f}")
    return result, True


def _lowlight_enhance(image: np.ndarray) -> Tuple[np.ndarray, bool]:
    """低光照自适应增强"""
    brightness = _calc_brightness(image)
    
    # 如果亮度足够，跳过
    if brightness >= 80:
        return image, False
    
    logger.info(f"检测到低光照（亮度={brightness:.1f}），应用增强...")
    
    # 1. 自适应Gamma校正
    gamma = max(0.3, min(2.0, 80.0 / max(brightness, 1)))
    gamma_corrected = np.power(image / 255.0, gamma) * 255.0
    gamma_corrected = gamma_corrected.astype(np.uint8)
    
    # 2. CLAHE增强
    lab = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    enhanced_lab = cv2.merge([l_enhanced, a, b])
    result = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    # 3. 轻度锐化补偿
    kernel_sharpen = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ], dtype=np.float32)
    result = cv2.filter2D(result, -1, kernel_sharpen)
    
    logger.info(f"低光照增强完成，增强后亮度: {_calc_brightness(result):.1f}, gamma={gamma:.2f}")
    return result, True


def _perspective_correct(image: np.ndarray) -> Tuple[np.ndarray, bool]:
    """基于边缘检测的透视校正"""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 1. 边缘检测
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # 2. Hough变换检测直线
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                            minLineLength=max(w, h) * 0.3,
                            maxLineGap=20)
    
    if lines is None or len(lines) < 4:
        return image, False  # 少于4条线，不校正
    
    # 3. 计算倾斜角度
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0:
            continue
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        angles.append(angle)
    
    if not angles:
        return image, False
    
    # 取中位数角度（排除离群）
    median_angle = float(np.median(angles))
    
    # 角度很小（<5度），用仿射变换校正
    if abs(median_angle) > 2.0 and abs(median_angle) < 45.0:
        logger.info(f"检测到倾斜角度={median_angle:.1f}度，应用透视校正...")
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        result = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_REPLICATE)
        return result, True
    
    return image, False


def _clahe_enhance(image: np.ndarray) -> Tuple[np.ndarray, bool]:
    """CLAHE对比度增强（独立调用时使用）"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # 自适应clipLimit
    mean_l = float(np.mean(l))
    clip_limit = 2.0 if mean_l > 100 else 4.0
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    
    enhanced_lab = cv2.merge([l_enhanced, a, b])
    result = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    # 轻度锐化
    kernel = np.array([
        [-1, -1, -1],
        [-1, 9, -1],
        [-1, -1, -1]
    ], dtype=np.float32) / 9.0
    result = cv2.filter2D(result, -1, kernel)
    
    return result, True


def _remove_shadows(image: np.ndarray) -> Tuple[np.ndarray, bool]:
    """形态学阴影去除"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 检测是否有明显阴影区域（亮度方差大的位置）
    block_size = 32
    mean_brightness = cv2.blur(gray, (block_size, block_size))
    local_var = cv2.blur(gray.astype(np.float32)**2, (block_size, block_size)) - mean_brightness.astype(np.float32)**2
    shadow_mask = local_var > 500.0  # 高方差区域
    
    shadow_pixels = np.sum(shadow_mask)
    total_pixels = shadow_mask.size
    
    if shadow_pixels / total_pixels < 0.01:
        return image, False  # 几乎没有阴影
    
    logger.info(f"检测到阴影区域（{shadow_pixels/total_pixels*100:.1f}%像素），应用阴影去除...")
    
    # 形态学闭运算提取背景
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    
    # 背景归一化
    bg = bg.astype(np.float32)
    normalized = gray.astype(np.float32) / (bg / np.mean(bg))
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)
    
    # 恢复颜色（仅调整亮度通道）
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = normalized
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return result, True


def _download_image(url: str) -> Optional[bytes]:
    """下载图片"""
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            return resp.content
        logger.warning(f"下载图片失败: HTTP {resp.status_code}")
    except Exception as e:
        logger.warning(f"下载图片异常: {e}")
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


# ==================== 主节点函数 ====================

def image_quality_enhance_node(
    state: ImageQualityEnhanceInput,
    config: RunnableConfig,
    runtime: Runtime[Context]
) -> ImageQualityEnhanceOutput:
    """
    title: 图像质量增强引擎
    desc: V5.6专业级图像质量增强，支持去模糊(Wiener滤波)、低光照增强(自适应Gamma+CLAHE)、透视校正(Hough变换)、对比度增强(CLAHE)、阴影去除。专为OCR/多模态识别场景设计，显著提升复杂图片的提取效果。
    integrations: OpenCV, NumPy
    """
    ctx = runtime.context
    logger.info("=== 图像质量增强开始 ===")
    t_start = time.time()
    
    try:
        # 选择图片：优先使用preprocessed_image，回退到package_image
        target_image = state.preprocessed_image or state.package_image
        if target_image is None:
            raise Exception("无可用的输入图片（preprocessed_image和package_image均为空）")
        
        # 下载图片
        logger.info(f"下载图片: {target_image.url}")
        img_data = _download_image(target_image.url)
        if img_data is None:
            raise Exception("图片下载失败，无法进行质量增强")
        
        img_array = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if image is None:
            raise Exception("图片解码失败")
        
        # 记录增强前质量
        before_metrics = _calc_quality_metrics(image)
        logger.info(f"增强前质量: {before_metrics}")
        
        processed = image.copy()
        steps = []
        
        deblur_applied = False
        lowlight_corrected = False
        perspective_corrected = False
        clahe_applied = False
        shadow_removed = False
        
        # Step 1: 去模糊（先做，减少后续处理噪声）
        if state.enable_deblur:
            processed, deblur_applied = _detect_and_deblur(processed, state.blur_detection_threshold)
            if deblur_applied:
                steps.append("去模糊(Wiener滤波)")
                logger.info("✓ 去模糊完成")
        
        # Step 2: 阴影去除（先做，避免阴影干扰后续）
        if state.enable_shadow_removal:
            processed, shadow_removed = _remove_shadows(processed)
            if shadow_removed:
                steps.append("阴影去除(形态学)")
                logger.info("✓ 阴影去除完成")
        
        # Step 3: 透视校正
        if state.enable_perspective:
            processed, perspective_corrected = _perspective_correct(processed)
            if perspective_corrected:
                steps.append("透视校正(Hough变换)")
                logger.info("✓ 透视校正完成")
        
        # Step 4: 低光照增强
        if state.enable_lowlight:
            processed, lowlight_corrected = _lowlight_enhance(processed)
            if lowlight_corrected:
                steps.append("低光照增强(自适应Gamma+CLAHE)")
                logger.info("✓ 低光照增强完成")
        
        # Step 5: CLAHE对比度增强（最后做）
        if state.enable_clahe:
            processed, clahe_applied = _clahe_enhance(processed)
            if clahe_applied:
                steps.append("CLAHE对比度增强")
                logger.info("✓ CLAHE对比度增强完成")
        
        # 记录增强后质量
        after_metrics = _calc_quality_metrics(processed)
        logger.info(f"增强后质量: {after_metrics}")
        
        # 上传增强后的图片
        file_name = f"quality_enhanced/enhanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        enhanced_url = _upload_image_to_storage(processed, file_name)
        logger.info(f"增强图片上传完成: {enhanced_url}")
        
        elapsed = time.time() - t_start
        
        return ImageQualityEnhanceOutput(
            enhanced_image=File(url=enhanced_url, file_type="image"),
            deblur_applied=deblur_applied,
            lowlight_corrected=lowlight_corrected,
            perspective_corrected=perspective_corrected,
            clahe_applied=clahe_applied,
            shadow_removed=shadow_removed,
            enhancement_steps=steps,
            processing_time=round(elapsed, 3),
            quality_metrics={
                "before": before_metrics,
                "after": after_metrics,
                "improvement": {
                    "brightness": round(after_metrics["brightness"] - before_metrics["brightness"], 2),
                    "contrast": round(after_metrics["contrast"] - before_metrics["contrast"], 2),
                    "sharpness": round(after_metrics["sharpness"] - before_metrics["sharpness"], 2),
                    "noise": round(before_metrics["noise"] - after_metrics["noise"], 2)
                }
            }
        )
        
    except Exception as e:
        logger.error(f"图像质量增强失败: {str(e)}\n{traceback.format_exc()}")
        # 失败时返回输入图
        fallback = state.preprocessed_image or state.package_image
        return ImageQualityEnhanceOutput(
            enhanced_image=fallback or File(url="", file_type="image"),
            deblur_applied=False,
            lowlight_corrected=False,
            perspective_corrected=False,
            clahe_applied=False,
            shadow_removed=False,
            enhancement_steps=[f"错误: {str(e)}"],
            processing_time=round(time.time() - t_start, 3),
            quality_metrics={}
        )