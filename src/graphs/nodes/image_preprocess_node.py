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

from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context
from graphs.state import ImagePreprocessInput, ImagePreprocessOutput
from utils.file.file import File
from coze_coding_dev_sdk.s3 import S3SyncStorage

logger = logging.getLogger(__name__)


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
    """下载图片并转为OpenCV格式"""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        resp = requests.get(url, timeout=30, headers=headers)
        resp.raise_for_status()
        pil_img = Image.open(BytesIO(resp.content))
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
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

    # 1. 质量评估
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

    # 2. 质量分级跳过预处理：高质量图（score > 70）直接返回原图
    #    大幅提升识别速度，避免过度处理
    if quality_score >= 70:
        logger.info(f"图片质量良好(score={quality_score})，跳过预处理增强")
        info["enhanced"] = False
        info["skip_reason"] = "quality_good"
        info["output_size"] = img.shape[:2]
        return img, info

    # 3. 自适应增强（仅中低质量图需要）
    result = adaptive_enhance(img, strategy)

    info["enhanced"] = True
    info["output_size"] = result.shape[:2]

    return result, info


def image_preprocess_node(
    state: ImagePreprocessInput,
    config: RunnableConfig,
    runtime: Runtime[Context]
) -> ImagePreprocessOutput:
    """
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
