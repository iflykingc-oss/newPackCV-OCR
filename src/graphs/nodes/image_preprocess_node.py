<<<<<<< HEAD
#!/usr/bin/env python3
"""图片预处理节点 - 文件验证和基础处理"""
import os
import logging
=======
# -*- coding: utf-8 -*-
"""
图像预处理节点 - 深度优化版 V2.5
自适应图像质量评估 + 多策略预处理管线
1. 质量评估：模糊检测/亮度/对比度/文本密度
2. 自适应策略选择：普通/暗图/模糊/低对比度/高对比度
3. CLAHE对比度增强 + 倾斜校正 + Unsharp锐化
"""
import os
import time
import logging
import requests
from typing import Optional, Dict, Any, Tuple, List
from io import BytesIO

import cv2
import numpy as np
from PIL import Image

>>>>>>> origin/main
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context
from graphs.state import ImagePreprocessInput, ImagePreprocessOutput
<<<<<<< HEAD
from utils.file.file import File, FileOps
=======
from utils.file.file import File
from coze_coding_dev_sdk.s3 import S3SyncStorage
>>>>>>> origin/main

logger = logging.getLogger(__name__)


<<<<<<< HEAD
=======
def _get_s3_storage() -> S3SyncStorage:
    """获取S3对象存储客户端"""
    return S3SyncStorage(
        endpoint_url=os.getenv("COZE_BUCKET_ENDPOINT_URL"),
        access_key="",
        secret_key="",
        bucket_name=os.getenv("COZE_BUCKET_NAME"),
        region="cn-beijing",
    )


def download_image(url: str) -> Optional[np.ndarray]:
    """下载图片并转为OpenCV格式，支持多种图片格式转换"""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        resp = requests.get(url, timeout=30, headers=headers)
        resp.raise_for_status()

        # Direction ⑤：多格式兼容 — 尝试用PIL识别并转换所有主流格式
        raw_bytes = BytesIO(resp.content)
        try:
            pil_img = Image.open(raw_bytes)
            # 自动转换RGBA/CMYK/P等非RGB模式
            if pil_img.mode in ('RGBA', 'LA', 'PA'):
                # 透明背景填充白色
                bg = Image.new('RGB', pil_img.size, (255, 255, 255))
                if pil_img.mode == 'RGBA':
                    bg.paste(pil_img, mask=pil_img.split()[3])
                else:
                    pil_img = pil_img.convert('RGBA')
                    bg.paste(pil_img, mask=pil_img.split()[3])
                pil_img = bg
            elif pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')

            # 限制最大尺寸（防止超大图片导致OOM）
            max_dim = 4096
            if pil_img.width > max_dim or pil_img.height > max_dim:
                scale = max_dim / max(pil_img.width, pil_img.height)
                new_w = int(pil_img.width * scale)
                new_h = int(pil_img.height * scale)
                pil_img = pil_img.resize((new_w, new_h), getattr(Image, 'LANCZOS', getattr(Image, 'ANTIALIAS', 1)))
                logger.info(f"图片缩放至 {(new_w, new_h)}（原尺寸过大）")

            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception as pil_err:
            logger.warning(f"PIL解析失败({pil_err})，尝试兜底")
            # 兜底：直接按JPEG解码
            return cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        logger.warning(f"图片下载失败: {e}")
        return None


# ==================== 图像质量评估 ====================

def assess_image_quality(img: np.ndarray) -> Dict[str, Any]:
    """
    多维图像质量评估
    返回质量报告，指导自适应预处理策略选择
    """
    report: Dict[str, Any] = {
        "blurry": False,
        "too_dark": False,
        "low_contrast": False,
        "high_contrast": False,
        "quality_score": 100.0,
        "recommended_strategy": "normal"
    }

    if img is None or img.size == 0:
        report["quality_score"] = 0
        return report

    h, w = img.shape[:2]
    report["dimensions"] = {"width": w, "height": h}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. 模糊检测（Laplacian方差）
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    report["laplacian_var"] = round(laplacian_var, 2)
    if laplacian_var < 30:
        report["blurry"] = True
        report["quality_score"] -= 30
    elif laplacian_var < 60:
        report["blurry"] = False
        report["quality_score"] -= 10

    # 2. 亮度检测
    mean_brightness = np.mean(gray)
    report["mean_brightness"] = round(mean_brightness, 2)
    if mean_brightness < 40:
        report["too_dark"] = True
        report["quality_score"] -= 25
    elif mean_brightness < 80:
        report["too_dark"] = False
        report["quality_score"] -= 10

    # 3. 对比度检测
    contrast = np.std(gray)
    report["contrast"] = round(contrast, 2)
    if contrast < 30:
        report["low_contrast"] = True
        report["quality_score"] -= 20
    elif contrast > 80:
        report["high_contrast"] = True
        report["quality_score"] -= 5

    # 4. 边缘密度（文本区域占比）
    try:
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.sum(edges > 0)) / (h * w)
        report["edge_density"] = round(edge_density, 5)
        if edge_density < 0.001:
            report["quality_score"] -= 15  # 几乎无文本
        elif edge_density < 0.01:
            report["quality_score"] -= 5   # 少量文本
    except Exception:
        report["edge_density"] = 0.0

    # 5. 自适应策略选择
    if report["too_dark"] and report["low_contrast"]:
        report["recommended_strategy"] = "dark_low_contrast"
    elif report["blurry"] and report["low_contrast"]:
        report["recommended_strategy"] = "blurry"
    elif report["low_contrast"]:
        report["recommended_strategy"] = "low_contrast"
    elif report["too_dark"]:
        report["recommended_strategy"] = "dark"
    elif report["high_contrast"]:
        report["recommended_strategy"] = "high_contrast"
    else:
        report["recommended_strategy"] = "normal"

    report["quality_score"] = max(0, min(100, report["quality_score"]))
    return report


def adaptive_enhance(img: np.ndarray, strategy: str = "normal") -> np.ndarray:
    """
    根据质量评估策略选择不同的预处理管线
    增强版：去噪 + 形态学文本增强 + 多级CLAHE
    """
    if img is None or img.size == 0:
        return img

    result = img.copy()
    h, w = result.shape[:2]

    # ====== 极低质量图超分辨率上采样 ======
    # 当图片较小或非常模糊时，通过上采样增加文字像素密度
    try:
        gray_check = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray_check, cv2.CV_64F).var()
        contrast_val = np.std(gray_check)
        min_dim = min(h, w)
        
        # 上采样触发条件：小图 或 模糊 或 低对比度
        needs_upscale = False
        upscale_factor = 1.0
        
        if min_dim < 800:
            needs_upscale = True
            upscale_factor = max(1.5, 1600.0 / min_dim)
        elif lap_var < 40 and min_dim < 1200:
            needs_upscale = True
            upscale_factor = 1.5
        elif contrast_val < 15:
            needs_upscale = True
            upscale_factor = 1.5
        
        # 限制最大上采样倍数
        upscale_factor = min(upscale_factor, 3.0)
        
        if needs_upscale and upscale_factor > 1.1:
            new_h = int(h * upscale_factor)
            new_w = int(w * upscale_factor)
            result = cv2.resize(result, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            h, w = result.shape[:2]
            logger.info(f"低质量图上采样: orig={int(h/upscale_factor)}x{int(w/upscale_factor)} -> {w}x{h} (factor={upscale_factor:.1f}, lap={lap_var:.0f}, contrast={contrast_val:.0f})")
    except Exception as e:
        logger.warning(f"上采样失败: {e}")

    # 大图缩放（所有策略通用）
    max_dimension = 2000
    if max(h, w) > max_dimension:
        scale = max_dimension / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        result = cv2.resize(result, (new_w, new_h), interpolation=cv2.INTER_AREA)
        h, w = result.shape[:2]

    # ====== Direction ②：小字区域检测与增强 ======
    try:
        gray_check = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        # 形态学梯度检测边缘密度
        grad_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        gradient = cv2.morphologyEx(gray_check, cv2.MORPH_GRADIENT, grad_kernel)
        # 二值化边缘
        _, edge_binary = cv2.threshold(gradient, 20, 255, cv2.THRESH_BINARY)

        # 计算小尺度边缘占比（用较小结构元素检测细节）
        detail_density = np.sum(edge_binary > 0) / (h * w)

        # 文字尺寸估算：检测轮廓的平均高度
        contours, _ = cv2.findContours(edge_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        text_heights = []
        for cnt in contours:
            _, _, _, ch = cv2.boundingRect(cnt)
            if 4 < ch < 30:  # 小到中等文字高度
                text_heights.append(ch)
        avg_text_h = np.mean(text_heights) if text_heights else 0

        # 小字判断：平均文字高度 < 12px 或 细节密度 > 15%
        has_small_text = avg_text_h < 12 and detail_density > 0.08

        if has_small_text and avg_text_h > 4:
            # 对小字图片做二次放大（针对小字场景专用）
            small_text_scale = max(1.5, min(3.0, 16.0 / max(avg_text_h, 1)))
            if small_text_scale > 1.2:
                new_h = int(h * small_text_scale)
                new_w = int(w * small_text_scale)
                result = cv2.resize(result, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                h, w = result.shape[:2]
                logger.info(f"小字增强: upscale={small_text_scale:.1f}x, avg_h={avg_text_h:.0f}px, density={detail_density:.2f}")

        # 对小字场景额外用自适应二值化增强对比度
        if has_small_text:
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            # 小核自适应阈值（对小文字更敏感）
            block_size = max(11, 15 if avg_text_h > 8 else 11)
            if block_size % 2 == 0:
                block_size += 1
            adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY, block_size, 4)
            # 与原图融合（保留颜色信息但增强文字边缘）
            color_enhanced = cv2.addWeighted(result, 0.6, cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR), 0.4, 0)
            result = color_enhanced
    except Exception as e:
        logger.warning(f"小字增强跳过: {e}")

    # ====== 步骤1：去噪（预处理核心，所有策略通用）======
    try:
        # 轻度双边滤波保边去噪，不损失边缘
        result = cv2.bilateralFilter(result, d=5, sigmaColor=30, sigmaSpace=30)
        # 对特别差的图片再加一次快速去噪
        if strategy in ("dark_low_contrast", "blurry", "dark"):
            result = cv2.fastNlMeansDenoisingColored(result, None, h=10, hColor=10, 
                                                      templateWindowSize=7, searchWindowSize=21)
    except Exception as e:
        logger.warning(f"去噪步骤失败: {e}")

    # ====== 步骤2：形态学文本增强 ======
    try:
        if strategy in ("blurry", "dark", "dark_low_contrast"):
            # 对模糊/暗图：先提取边缘，再增强文本区域
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            # 使用形态学梯度增强文字边缘
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
            # 合并梯度图与原图
            result = cv2.addWeighted(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), 0.7,
                                      cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR), 0.3, 0)
    except Exception as e:
        logger.warning(f"形态学增强失败: {e}")

    # 倾斜校正（所有策略通用）
    try:
        result, _ = _detect_and_correct_skew(result)
    except Exception:
        pass

    if strategy == "dark" or strategy == "dark_low_contrast":
        # 暗图策略：强伽马校正 → CLAHE → 锐化
        try:
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            gamma = 0.6 if strategy == "dark_low_contrast" else 0.7
            gamma_correction = np.power(gray / 255.0, gamma).astype(np.float32) * 255.0
            gamma_correction = np.clip(gamma_correction, 0, 255).astype(np.uint8)
            # 伽马后再CLAHE
            clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
            enhanced_l = clahe.apply(gamma_correction)
            result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            # 用伽马+CLAHE结果替换原图亮度
            result = cv2.cvtColor(enhanced_l, cv2.COLOR_GRAY2BGR)
        except Exception as e:
            logger.warning(f"暗图增强策略失败: {e}")

    elif strategy == "blurry":
        # 模糊策略：强锐化 → CLAHE + 自适应阈值
        try:
            # 强拉普拉斯锐化
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            result = cv2.filter2D(result, -1, kernel)
            # Unsharp Masking二次锐化
            gaussian = cv2.GaussianBlur(result, (0, 0), 1.5)
            result = cv2.addWeighted(result, 1.8, gaussian, -0.8, 0)
            # CLAHE
            lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l)
            result = cv2.cvtColor(cv2.merge([l_enhanced, a, b]), cv2.COLOR_LAB2BGR)
        except Exception as e:
            logger.warning(f"模糊增强策略失败: {e}")

    elif strategy == "low_contrast" or strategy == "dark_low_contrast":
        # 低对比度策略：强CLAHE + 对比度拉伸
        try:
            lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            # 自适应clipLimit
            contrast_val = np.std(l)
            clip_limit = 4.0 if contrast_val < 20 else 3.0
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l)
            # 对比度拉伸
            min_val, max_val = np.percentile(l_enhanced, (5, 95))
            if max_val > min_val:
                l_stretched = np.clip((l_enhanced.astype(np.float32) - min_val) * 
                                       255.0 / (max_val - min_val), 0, 255).astype(np.uint8)
            else:
                l_stretched = l_enhanced
            result = cv2.cvtColor(cv2.merge([l_stretched, a, b]), cv2.COLOR_LAB2BGR)
        except Exception as e:
            logger.warning(f"低对比度增强策略失败: {e}")

    elif strategy == "high_contrast":
        try:
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray_eq = clahe.apply(gray)
            result = cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2BGR)
        except Exception as e:
            logger.warning(f"高对比度增强策略失败: {e}")

    # ====== 标准CLAHE（所有策略通用）======
    try:
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        # 根据对比度自适应clipLimit
        contrast_val = np.std(l)
        clip_limit = 3.0 if contrast_val < 30 else 2.0
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    except Exception as e:
        logger.warning(f"标准CLAHE增强失败: {e}")

    # ====== 自适应阈值二值化（仅对极差图片）======
    if strategy in ("dark_low_contrast", "blurry"):
        try:
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            # 自适应阈值生成二值化辅助通道
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 31, 5)
            # 混合原图和二值化结果（保持自然感的同时增强文字）
            result = cv2.addWeighted(result, 0.6, 
                                      cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR), 0.4, 0)
        except Exception as e:
            logger.warning(f"自适应二值化失败: {e}")

    # ====== 最终锐化（所有策略通用）======
    try:
        gaussian = cv2.GaussianBlur(result, (0, 0), 2.0)
        result = cv2.addWeighted(result, 1.5, gaussian, -0.5, 0)
    except Exception as e:
        logger.warning(f"锐化失败: {e}")

    return result


def _detect_and_correct_skew(img: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    检测并校正图像倾斜角度（适用于扫描件/拍照歪斜）
    通过霍夫变换检测直线角度，修正旋转
    """
    if img is None or img.size == 0:
        return img, 0.0

    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 边缘检测
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        # 霍夫变换检测直线
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

        if lines is None or len(lines) < 5:
            return img, 0.0  # 直线太少，不校正

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(x2 - x1) > 10:  # 忽略垂直线
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                angles.append(angle)

        if not angles:
            return img, 0.0

        # 取中位数角度
        median_angle = np.median(angles)
        # 小角度校正（只校正|角度|在0.5-45度之间）
        if abs(median_angle) < 0.5 or abs(median_angle) > 45:
            return img, 0.0

        logger.info(f"检测到图像倾斜: {median_angle:.2f}度，正在进行校正")

        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(img, rotation_matrix, (w, h),
                                 flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated, median_angle

    except Exception as e:
        logger.warning(f"倾斜校正失败: {e}")
        return img, 0.0


def _perspective_correction(img: np.ndarray) -> np.ndarray:
    """
    透视校正：针对曲面包装（瓶身/罐体/弧形标签）
    检测图像中的四边形轮廓，做透视变换展平
    适用于洗发水瓶、酸奶瓶、酱料瓶等曲面包装
    """
    if img is None or img.size == 0:
        return img
    
    try:
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. 检查是否有明显的垂直方向扭曲（瓶子/罐体特征）
        # 方法：检测图像中心区域的垂直边缘密集度
        center_region = gray[:, w//4:3*w//4]
        
        # 2. Canny边缘检测
        edges = cv2.Canny(center_region, 30, 100)
        
        # 3. 检测垂直线条（瓶身竖直边缘）
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, 
                                minLineLength=int(h*0.3), maxLineGap=20)
        
        if lines is None or len(lines) < 4:
            return img  # 线条不足，非透视场景
        
        # 4. 分离水平和垂直线条
        v_lines = []  # 垂直线
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if angle > 60:  # 接近垂直
                v_lines.append(((x1 + x2) // 2, (y1 + y2) // 2))
        
        if len(v_lines) < 4:
            return img  # 垂直线不足
        
        # 5. 估算透视变换
        # 找到图像左右两侧的关键点
        v_lines.sort(key=lambda p: p[0])  # 按x坐标排序
        
        # 取左右两侧最具代表性的点
        left_x = np.mean([p[0] for p in v_lines[:len(v_lines)//4]]) if v_lines else w//4
        right_x = np.mean([p[0] for p in v_lines[-len(v_lines)//4:]]) if v_lines else 3*w//4
        
        # 检测顶部和底部的水平偏移
        top_lines = [p for p in v_lines if p[1] < h//3]
        bot_lines = [p for p in v_lines if p[1] > 2*h//3]
        
        if len(top_lines) > 2 and len(bot_lines) > 2:
            top_left = np.mean([p[0] for p in top_lines[:len(top_lines)//2]])
            top_right = np.mean([p[0] for p in top_lines[-len(top_lines)//2:]])
            bot_left = np.mean([p[0] for p in bot_lines[:len(bot_lines)//2]])
            bot_right = np.mean([p[0] for p in bot_lines[-len(bot_lines)//2:]])
            
            # 检查是否有梯形变形（透视征兆）
            top_width = top_right - top_left
            bot_width = bot_right - bot_left
            
            if top_width > 0 and bot_width > 0:
                width_ratio = top_width / bot_width if bot_width > top_width else bot_width / top_width
                
                # 如果上下宽度差异超过15%，认为是透视变形
                if width_ratio > 1.15:
                    logger.info(f"检测到透视变形: top_w={top_width:.0f}, bot_w={bot_width:.0f}, ratio={width_ratio:.2f}")
                    
                    # 计算透视变换源点和目标点
                    margin = 20
                    src_pts = np.float32([
                        [max(0, top_left - margin), margin],           # 左上
                        [min(w, top_right + margin), margin],           # 右上
                        [max(0, bot_left - margin), h - margin],       # 左下
                        [min(w, bot_right + margin), h - margin]        # 右下
                    ])
                    
                    dst_w = int(max(top_width, bot_width)) + 2 * margin
                    dst_pts = np.float32([
                        [0, 0],
                        [dst_w, 0],
                        [0, h],
                        [dst_w, h]
                    ])
                    
                    # 计算透视变换矩阵
                    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
                    corrected = cv2.warpPerspective(img, matrix, (dst_w, h),
                                                    flags=cv2.INTER_CUBIC,
                                                    borderMode=cv2.BORDER_REPLICATE)
                    logger.info(f"透视校正完成: {w}x{h} -> {corrected.shape[1]}x{corrected.shape[0]}")
                    return corrected
    
    except Exception as e:
        logger.warning(f"透视校正失败: {e}")
    
    return img


def enhance_for_ocr(img: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    自适应OCR预处理管线
    1. 质量评估 → 2. 自适应策略选择 → 3. 增强处理
    """
    info: Dict[str, Any] = {
        "original_size": img.shape[:2] if img is not None else (0, 0),
        "enhanced": False,
        "resized": False
    }

    if img is None or img.size == 0:
        return img, info

    # 1. 透视校正（曲面包装展平）：先于质量评估，因为透视变形本身不算"低质量"
    corrected = _perspective_correction(img)
    if corrected.shape != img.shape:
        logger.info(f"透视校正生效: {img.shape} -> {corrected.shape}")
        img = corrected
        info["perspective_corrected"] = True
    else:
        info["perspective_corrected"] = False

    # 2. 质量评估
    quality_report = assess_image_quality(img)
    strategy = quality_report.get("recommended_strategy", "normal")
    quality_score = quality_report.get("quality_score", 50)
    info["quality_report"] = quality_report
    info["strategy"] = strategy
    info["quality_score"] = quality_score
    logger.info(f"图像质量评估: score={quality_score}, "
                f"strategy={strategy}, blurry={quality_report['blurry']}, "
                f"dark={quality_report['too_dark']}, "
                f"contrast={quality_report['contrast']}")

    # 3. 倾斜校正（补充透视校正后的旋转修正）
    corrected_img, angle = _detect_and_correct_skew(img)
    if abs(angle) > 0.5:
        img = corrected_img
        info["rotation_angle"] = round(angle, 2)
        logger.info(f"倾斜校正: {angle:.2f}度")

    # 4. 质量分级跳过预处理：高质量图（score > 70）直接返回
    if quality_score >= 70:
        logger.info(f"图片质量良好(score={quality_score})，跳过预处理增强")
        info["enhanced"] = False
        info["skip_reason"] = "quality_good"
        info["output_size"] = img.shape[:2]
        return img, info

    # 5. 自适应增强（仅中低质量图需要）
    result = adaptive_enhance(img, strategy)

    info["enhanced"] = True
    info["output_size"] = result.shape[:2]

    return result, info


>>>>>>> origin/main
def image_preprocess_node(
    state: ImagePreprocessInput,
    config: RunnableConfig,
    runtime: Runtime[Context]
) -> ImagePreprocessOutput:
    """
<<<<<<< HEAD
    title: 文件预处理
    desc: 对输入文件进行验证和基础预处理，确保文件可被后续节点正确处理
    integrations: 无
    """
    ctx = runtime.context
    
    input_file = state.input_file
    
    # 文件类型判断
    file_url = input_file.url
    image_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif']
    document_extensions = ['.pdf', '.docx', '.doc', '.pptx', '.ppt', '.xlsx', '.xls', '.txt', '.md']
    
    file_ext = os.path.splitext(file_url)[1].lower() if file_url else ''
    is_image = file_ext in image_extensions
    
    # 预处理逻辑：
    # 1. 验证文件可访问性（通过FileOps读取）
    # 2. 图片文件：标记为已预处理（后续可进行增强处理）
    # 3. 文档文件：验证格式支持，准备OCR处理
    
    try:
        # 验证文件可读
        content = FileOps.read_bytes(input_file)
        if len(content) == 0:
            logger.warning(f"文件内容为空: {file_url}")
            
        # 预处理完成，返回验证后的文件
        preprocessed_file = input_file
        
        file_type_str = "图片" if is_image else "文档"
        logger.info(f"预处理完成: {file_url} (类型: {file_type_str}, 大小: {len(content)} bytes)")
        
    except Exception as e:
        logger.error(f"文件预处理失败: {e}")
        # 即使失败也返回原文件，让后续节点处理错误
        preprocessed_file = input_file
    
    return ImagePreprocessOutput(preprocessed_file=preprocessed_file)
=======
    title: 图像预处理
    desc: 对输入图片进行OCR前轻量级预处理（CLAHE对比度增强+锐化），处理后的图片上传到对象存储。
          RapidOCR自带文本检测+方向分类，无需额外方向/区域检测。
    integrations: 对象存储
    """
    ctx = runtime.context
    start_time = time.time()

    try:
        img_url = ""
        if state.package_image and state.package_image.url:
            img_url = state.package_image.url

        if not img_url:
            logger.error("无可用图片URL")
            return ImagePreprocessOutput(
                is_enhanced=False
            )

        img = download_image(img_url)
        if img is None:
            logger.error("图片下载失败")
            return ImagePreprocessOutput(
                is_enhanced=False
            )

        logger.info(f"图片下载成功: shape={img.shape}")

        # 轻量级图像增强
        enhanced_img, processing_info = enhance_for_ocr(img)
        logger.info(f"图像增强完成: {processing_info}")

        # 保存增强后的图片并上传S3（使用JPEG格式减小文件体积）
        success, img_encoded = cv2.imencode('.jpg', enhanced_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not success:
            logger.error("图片编码失败")
            return ImagePreprocessOutput(
                is_enhanced=False
            )

        img_bytes = img_encoded.tobytes()

        # 上传S3
        storage = _get_s3_storage()
        file_name = f"preprocessed/preprocessed_{int(time.time())}.jpg"
        key = storage.upload_file(
            file_content=img_bytes,
            file_name=file_name,
            content_type='image/jpeg'
        )
        presigned_url = storage.generate_presigned_url(key=key, expire_time=3600)
        logger.info(f"预处理图片已上传: {presigned_url[:60]}...")

        elapsed_time = time.time() - start_time
        processing_info["elapsed_time"] = round(elapsed_time, 2)
        logger.info(f"图像预处理完成，耗时: {elapsed_time:.2f}秒")

        is_rotated = processing_info.get("rotation_angle", 0) != 0

        return ImagePreprocessOutput(
            preprocessed_image=File(url=presigned_url, file_type="image"),
            is_rotated=is_rotated,
            is_enhanced=processing_info.get("enhanced", False),
            processing_info=processing_info
        )

    except Exception as e:
        logger.error(f"图像预处理异常: {e}", exc_info=True)
        return ImagePreprocessOutput(
            is_enhanced=False
        )
>>>>>>> origin/main
