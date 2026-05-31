# -*- coding: utf-8 -*-
"""
OCR文字识别节点 - 深度优化版 V2.3
基于RapidOCR（ONNX引擎）为主，Tesseract为备选的多引擎OCR
关键优化：
1. RapidOCR单例模式，避免重复初始化
2. PaddleOCR已禁用（OneDNN兼容性问题）
3. 大图自动缩放加速推理
4. 智能文本后处理
"""
import os
import time
import logging
import re
import requests
from typing import List, Dict, Any, Optional, Tuple
from io import BytesIO

import cv2
import numpy as np
from PIL import Image

from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context
from graphs.state import OCRRecognizeInput, OCRRecognizeOutput
from utils.file.file import File

logger = logging.getLogger(__name__)

# ==================== 引擎可用性检测 ====================

_RAPID_OCR_AVAILABLE = False
_TESSERACT_AVAILABLE = False

try:
    from rapidocr_onnxruntime import RapidOCR  # type: ignore[import-untyped]
    _RAPID_OCR_AVAILABLE = True
    logger.info("RapidOCR引擎可用")
except ImportError:
    logger.warning("RapidOCR不可用（rapidocr_onnxruntime未安装）")

try:
    import pytesseract  # type: ignore[import-untyped]
    pytesseract.get_tesseract_version()
    _TESSERACT_AVAILABLE = True
    logger.info("Tesseract引擎可用")
except Exception:
    logger.warning("Tesseract不可用（未安装或不在PATH中）")

# PaddleOCR已禁用 - 因OneDNN兼容性问题无法在当前环境运行
# 如需启用请在环境变量中设置 ENABLE_PADDLEOCR=1
_ENABLE_PADDLE_OCR = os.getenv("ENABLE_PADDLEOCR", "0") == "1"


# ==================== 引擎单例 ====================

_rapid_ocr_instance = None


def _get_rapid_ocr():
    """获取RapidOCR单例实例"""
    global _rapid_ocr_instance
    if _rapid_ocr_instance is None and _RAPID_OCR_AVAILABLE:
        try:
            _rapid_ocr_instance = RapidOCR()
            logger.info("RapidOCR实例初始化成功")
        except Exception as e:
            logger.warning(f"RapidOCR初始化失败: {e}")
    return _rapid_ocr_instance


# ==================== 图片下载 ====================

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


# ==================== 图像预处理（仅用于Tesseract） ====================

def preprocess_for_traditional_ocr(img: np.ndarray) -> np.ndarray:
    """传统OCR引擎预处理：灰度化 + CLAHE + 自适应二值化"""
    if img is None or img.size == 0:
        return img

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    binary = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 10
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return cleaned


# ==================== RapidOCR引擎 ====================

def ocr_with_rapid(img: np.ndarray) -> Tuple[str, float, List[Dict[str, Any]]]:
    """使用RapidOCR进行OCR识别"""
    if not _RAPID_OCR_AVAILABLE:
        return "", 0.0, []

    try:
        ocr = _get_rapid_ocr()
        if ocr is None:
            return "", 0.0, []

        result, elapse = ocr(img)

        if not result:
            return "", 0.0, []

        texts: List[str] = []
        confidences: List[float] = []
        regions: List[Dict[str, Any]] = []

        for item in result:
            bbox = item[0]
            text = str(item[1])
            conf = float(item[2])

            texts.append(text)
            confidences.append(conf)

            if isinstance(bbox, list) and len(bbox) >= 4:
                xs = [int(p[0]) for p in bbox if isinstance(p, (list, tuple)) and len(p) >= 2]
                ys = [int(p[1]) for p in bbox if isinstance(p, (list, tuple)) and len(p) >= 2]
                if xs and ys:
                    regions.append({
                        "text": text,
                        "confidence": conf,
                        "bbox": [min(xs), min(ys), max(xs), max(ys)],
                        "type": "text"
                    })
                else:
                    regions.append({"text": text, "confidence": conf, "bbox": [], "type": "text"})
            else:
                regions.append({"text": text, "confidence": conf, "bbox": [], "type": "text"})

        full_text = "\n".join(texts)
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

        logger.info(f"RapidOCR识别: {len(texts)}行, 平均置信度={avg_conf:.4f}, 耗时={elapse}")

        return full_text, avg_conf, regions

    except Exception as e:
        logger.warning(f"RapidOCR识别失败: {e}")
        return "", 0.0, []


# ==================== Tesseract引擎 ====================

def ocr_with_tesseract(img: np.ndarray, psm: int = 6, lang: str = "chi_sim+eng") -> Tuple[str, float]:
    """使用Tesseract进行OCR识别"""
    if not _TESSERACT_AVAILABLE:
        return "", 0.0

    try:
        processed = preprocess_for_traditional_ocr(img)
        if len(processed.shape) == 2:
            pil_img = Image.fromarray(processed)
        else:
            pil_img = Image.fromarray(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))

        config_str = f'--psm {psm} -l {lang} --oem 3'
        text = pytesseract.image_to_string(pil_img, config=config_str).strip()

        try:
            data = pytesseract.image_to_data(pil_img, config=config_str, output_type=pytesseract.Output.DICT)
            confs = [int(c) for c in data.get('conf', []) if int(c) > 0]
            avg_conf = sum(confs) / len(confs) / 100.0 if confs else 0.0
        except Exception:
            avg_conf = 0.5 if text else 0.0

        return text, avg_conf

    except Exception as e:
        logger.warning(f"Tesseract OCR失败: {e}")
        return "", 0.0


# ==================== 多引擎融合 ====================

def multi_engine_ocr(img: np.ndarray, time_budget: float = 30.0) -> Tuple[str, float, List[Dict[str, Any]], str]:
    """
    多引擎融合OCR（带时间预算）
    优先级: RapidOCR > Tesseract
    """
    start = time.time()

    # 第一优先级: RapidOCR（推荐引擎）
    if _RAPID_OCR_AVAILABLE:
        text, conf, regions = ocr_with_rapid(img)
        if text and conf >= 0.3:
            logger.info(f"RapidOCR识别成功: 文本长度={len(text)}, 置信度={conf:.4f}")
            return text, conf, regions, "rapidocr"

        if text:
            logger.info(f"RapidOCR低置信度({conf:.4f})但仍有结果，作为保底")
            rapid_text, rapid_conf, rapid_regions = text, conf, regions
        else:
            rapid_text, rapid_conf, rapid_regions = "", 0.0, []
    else:
        rapid_text, rapid_conf, rapid_regions = "", 0.0, []

    # 检查时间预算
    elapsed = time.time() - start
    if elapsed > time_budget * 0.6:
        logger.info(f"时间预算不足({elapsed:.1f}s)，跳过Tesseract")
        if rapid_text:
            return rapid_text, rapid_conf, rapid_regions, "rapidocr_low_conf"

    # 第二优先级: Tesseract
    if _TESSERACT_AVAILABLE:
        # 先尝试中英混合
        text, conf = ocr_with_tesseract(img, psm=6, lang="chi_sim+eng")
        if text:
            if rapid_text and rapid_conf >= conf:
                return rapid_text, rapid_conf, rapid_regions, "rapidocr"
            return text, conf, [], "tesseract_chi_sim+eng"

        # 再尝试纯英文
        text, conf = ocr_with_tesseract(img, psm=6, lang="eng")
        if text:
            if rapid_text and rapid_conf >= conf:
                return rapid_text, rapid_conf, rapid_regions, "rapidocr"
            return text, conf, [], "tesseract_eng"

    # 返回最好的结果
    if rapid_text:
        return rapid_text, rapid_conf, rapid_regions, "rapidocr_low_conf"

    logger.warning("所有OCR引擎均未识别到文本")
    return "", 0.0, [], "none"


# ==================== 文本后处理 ====================

def post_process_ocr_text(text: str) -> str:
    """OCR文本后处理：去重、修复常见错误"""
    if not text:
        return text

    ocr_corrections = {"O": "0", "o": "0", "l": "1", "I": "1"}

    lines = text.split('\n')
    lines = [line.strip() for line in lines if line.strip()]

    # 去除重复行
    seen: List[str] = []
    unique_lines: List[str] = []
    for line in lines:
        if line not in seen:
            seen.append(line)
            unique_lines.append(line)

    # 修复日期/数字中的OCR错误
    processed_lines: List[str] = []
    for line in unique_lines:
        if re.search(r'\d{4}', line) and re.search(r'[年月日/-]', line):
            for wrong, correct in ocr_corrections.items():
                # 使用函数替换避免group reference错误
                line = re.sub(rf'(\d){re.escape(wrong)}', lambda m: m.group(1) + correct, line)
                line = re.sub(rf'{re.escape(wrong)}(\d)', lambda m: correct + m.group(1), line)
        processed_lines.append(line)

    return "\n".join(processed_lines)


# ==================== 主节点函数 ====================

def ocr_recognize_node(
    state: OCRRecognizeInput,
    config: RunnableConfig,
    runtime: Runtime[Context]
) -> OCRRecognizeOutput:
    """
    title: OCR文字识别
    desc: 多引擎融合OCR识别，优先使用RapidOCR(ONNX引擎)，备选Tesseract。
          支持自动引擎降级、原始图像回退、智能文本后处理。
    integrations: RapidOCR, Tesseract
    """
    ctx = runtime.context
    start_time = time.time()

    # 选择图片（优先预处理后的图片）
    image_url = ""
    if state.preprocessed_image and state.preprocessed_image.url:
        image_url = state.preprocessed_image.url
    elif state.package_image and state.package_image.url:
        image_url = state.package_image.url
    elif state.image and state.image.url:
        image_url = state.image.url

    if not image_url:
        logger.error("无可用图片")
        return OCRRecognizeOutput(
            raw_text="[ERROR] 无可用图片",
            ocr_confidence=0.0,
            engine_used="none",
            processing_time=time.time() - start_time
        )

    # 下载图片
    img = download_image(image_url)
    if img is None:
        return OCRRecognizeOutput(
            raw_text="[ERROR] 图片下载失败",
            ocr_confidence=0.0,
            engine_used="none",
            processing_time=time.time() - start_time
        )

    logger.info(f"图片下载成功: shape={img.shape}, dtype={img.dtype}")

    # 大图缩放（加速OCR推理）
    h, w = img.shape[:2]
    max_dim = 1500
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        logger.info(f"大图缩放: {w}x{h} -> {new_w}x{new_h}")

    # 多引擎融合OCR
    final_text, final_conf, ocr_regions, engine = multi_engine_ocr(img)

    # 如果预处理后图片识别为空，尝试原始图片
    if not final_text:
        original_url = ""
        if state.package_image and state.package_image.url:
            original_url = state.package_image.url
        elif state.image and state.image.url:
            original_url = state.image.url

        if original_url and original_url != image_url:
            logger.info("预处理图片OCR为空，尝试原始图片")
            original_img = download_image(original_url)
            if original_img is not None:
                oh, ow = original_img.shape[:2]
                if max(oh, ow) > max_dim:
                    o_scale = max_dim / max(oh, ow)
                    original_img = cv2.resize(original_img, (int(ow * o_scale), int(oh * o_scale)), interpolation=cv2.INTER_AREA)
                final_text, final_conf, ocr_regions, engine = multi_engine_ocr(original_img)
                if final_text:
                    engine = engine + "_original"

    # 文本后处理
    if final_text:
        final_text = post_process_ocr_text(final_text)

    elapsed_time = time.time() - start_time
    logger.info(f"OCR完成: 引擎={engine}, 耗时={elapsed_time:.2f}s, 文本长度={len(final_text)}, 置信度={final_conf:.4f}")

    return OCRRecognizeOutput(
        raw_text=final_text,
        ocr_confidence=final_conf,
        engine_used=engine,
        processing_time=elapsed_time,
        ocr_regions=ocr_regions
    )
