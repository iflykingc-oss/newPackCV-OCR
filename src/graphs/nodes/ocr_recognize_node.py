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

        # 智能后处理管线
        raw_text = "\n".join(texts)
        raw_conf = sum(confidences) / len(confidences) if confidences else 0.0

        full_text, avg_conf, processed_regions = post_process_ocr_results(texts, confidences, regions)

        logger.info(
            f"RapidOCR识别: {len(texts)}行→{len(processed_regions)}行后处理, "
            f"原始置信度={raw_conf:.4f}, 后处理置信度={avg_conf:.4f}, "
            f"耗时={elapse}"
        )

        return full_text, avg_conf, processed_regions

    except Exception as e:
        logger.warning(f"RapidOCR识别失败: {e}")
        return "", 0.0, []


# ==================== Tesseract引擎 ====================

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


def ocr_with_tesseract_multi_psm(img: np.ndarray, lang: str = "chi_sim+eng") -> Tuple[str, float]:
    """Tesseract多PSM扫描，尝试多个Page Segmentation Mode"""
    if not _TESSERACT_AVAILABLE:
        return "", 0.0

    best_text, best_conf = "", 0.0

    # PSM 6: 假设统一的文本块
    text, conf = ocr_with_tesseract(img, psm=6, lang=lang)
    if text and conf > best_conf:
        best_text, best_conf = text, conf

    # PSM 4: 假设单列可变字体
    if not best_text:
        text, conf = ocr_with_tesseract(img, psm=4, lang=lang)
        if text and conf > best_conf:
            best_text, best_conf = text, conf

    # PSM 3: 完全自动
    if not best_text:
        text, conf = ocr_with_tesseract(img, psm=3, lang=lang)
        if text and conf > best_conf:
            best_text, best_conf = text, conf

    return best_text, best_conf


# ==================== 多引擎融合 ====================

def multi_engine_ocr(img: np.ndarray, time_budget: float = 30.0) -> Tuple[str, float, List[Dict[str, Any]], str]:
    """
    多引擎融合OCR（带时间预算）
    优先级: RapidOCR > Tesseract多PSM > Tesseract英文
    融合策略：优先高置信度结果，RapidOCR低置信度时Tesseract兜底
    """
    start = time.time()

    # 第一优先级: RapidOCR（推荐引擎）
    rapid_text, rapid_conf, rapid_regions = "", 0.0, []
    if _RAPID_OCR_AVAILABLE:
        text, conf, regions = ocr_with_rapid(img)
        if text and conf >= 0.5:
            logger.info(f"RapidOCR高置信度: 文本长度={len(text)}, 置信度={conf:.4f}")
            return text, conf, regions, "rapidocr"
        rapid_text, rapid_conf, rapid_regions = text, conf, regions
        if text:
            logger.info(f"RapidOCR低置信度({conf:.4f}), 尝试Tesseract补充")

    # 检查时间预算
    elapsed = time.time() - start
    if elapsed > time_budget * 0.7:
        logger.info(f"时间预算不足({elapsed:.1f}s)，跳过Tesseract")
        if rapid_text:
            return rapid_text, rapid_conf, rapid_regions, "rapidocr_low_conf"

    # 第二优先级: Tesseract中英混合（多PSM）
    tesseract_text, tesseract_conf = "", 0.0
    if _TESSERACT_AVAILABLE:
        text, conf = ocr_with_tesseract_multi_psm(img, lang="chi_sim+eng")
        if text:
            tesseract_text, tesseract_conf = text, conf
            logger.info(f"Tesseract中英识别: 文本长度={len(text)}, 置信度={conf:.4f}")

    # 融合策略：选择置信度高的
    if rapid_text and tesseract_text:
        if rapid_conf >= tesseract_conf:
            logger.info(f"融合选择: RapidOCR(conf={rapid_conf:.4f}) > Tesseract(conf={tesseract_conf:.4f})")
            return rapid_text, rapid_conf, rapid_regions, "rapidocr"
        else:
            logger.info(f"融合选择: Tesseract(conf={tesseract_conf:.4f}) > RapidOCR(conf={rapid_conf:.4f})")
            return tesseract_text, tesseract_conf, [], "tesseract"
    elif rapid_text:
        return rapid_text, rapid_conf, rapid_regions, "rapidocr_low_conf"
    elif tesseract_text:
        return tesseract_text, tesseract_conf, [], "tesseract"

    # 第三优先级: Tesseract纯英文
    if _TESSERACT_AVAILABLE:
        text, conf = ocr_with_tesseract_multi_psm(img, lang="eng")
        if text:
            logger.info(f"Tesseract英文识别: 文本长度={len(text)}, 置信度={conf:.4f}")
            return text, conf, [], "tesseract_eng"

    # 最终保底
    if rapid_text:
        return rapid_text, rapid_conf, rapid_regions, "rapidocr_low_conf"

    logger.warning("所有OCR引擎均未识别到文本")
    return "", 0.0, [], "none"


# ==================== 文本后处理 ====================

def _box_iou(box_a: List[int], box_b: List[int]) -> float:
    """计算两个bbox的IoU (Intersection over Union)"""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / float(area_a + area_b - inter + 1e-6)


def _sort_boxes_reading_order(regions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """将文本区域按阅读顺序排序（先从上到下，再从左到右）
    参考 RapidOCR `sorted_boxes()` 实现，适配包装标签"""
    if not regions:
        return regions

    # 按bbox垂直中心y坐标分组（同行文本y中心差 < 行高的一半）
    for region in regions:
        bbox = region.get("bbox", [])
        if len(bbox) == 4:
            region["_cy"] = (bbox[1] + bbox[3]) / 2.0
            region["_cx"] = (bbox[0] + bbox[2]) / 2.0
            region["_h"] = bbox[3] - bbox[1]
        else:
            region["_cy"] = 0
            region["_cx"] = 0
            region["_h"] = 0

    # 按垂直中心y排序
    sorted_regions = sorted(regions, key=lambda r: r.get("_cy", 0))

    # 分组聚类：y中心差距 < 行高的一半视为同一行
    lines: List[List[Dict[str, Any]]] = []
    current_line: List[Dict[str, Any]] = []
    last_cy = -1000

    for region in sorted_regions:
        cy = region.get("_cy", 0)
        h = max(region.get("_h", 1), 1)
        if current_line and (cy - last_cy) > h * 0.5:
            # 新行，先排序当前行的左到右
            current_line.sort(key=lambda r: r.get("_cx", 0))
            lines.append(current_line)
            current_line = [region]
        else:
            current_line.append(region)
        last_cy = cy

    if current_line:
        current_line.sort(key=lambda r: r.get("_cx", 0))
        lines.append(current_line)

    # 展平
    result = []
    for line in lines:
        result.extend(line)

    # 清理临时字段
    for region in result:
        region.pop("_cy", None)
        region.pop("_cx", None)
        region.pop("_h", None)

    return result


def _merge_nearby_regions(regions: List[Dict[str, Any]], iou_thresh: float = 0.3) -> List[Dict[str, Any]]:
    """合并IoU过高（重叠）的重复区域，保留置信度最高的"""
    if not regions:
        return regions

    merged = []
    used = [False] * len(regions)

    for i, region_a in enumerate(regions):
        if used[i]:
            continue
        best = region_a
        for j, region_b in enumerate(regions):
            if i == j or used[j]:
                continue
            bbox_a = best.get("bbox", [])
            bbox_b = region_b.get("bbox", [])
            if len(bbox_a) == 4 and len(bbox_b) == 4:
                iou = _box_iou(bbox_a, bbox_b)
                if iou > iou_thresh:
                    # 保留置信度高的
                    if region_b.get("confidence", 0) > best.get("confidence", 0):
                        best = region_b
                    used[j] = True
        merged.append(best)
        used[i] = True

    return merged


def _deduplicate_ocr_texts(regions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """智能去重：对相同文本区域保留高置信度结果"""
    if not regions:
        return regions

    # 按文本内容分组
    text_groups: Dict[str, List[Dict[str, Any]]] = {}
    for region in regions:
        text = region.get("text", "").strip()
        if text:
            if text not in text_groups:
                text_groups[text] = []
            text_groups[text].append(region)

    # 对每个组保留置信度最高的
    result = []
    for text, group in text_groups.items():
        if len(group) == 1:
            result.append(group[0])
        else:
            # IoU过滤: 相互重叠的只取一个
            seen: List[Dict[str, Any]] = []
            for region in group:
                is_duplicate = False
                for existing in seen:
                    bbox_a = region.get("bbox", [])
                    bbox_b = existing.get("bbox", [])
                    if len(bbox_a) == 4 and len(bbox_b) == 4:
                        if _box_iou(bbox_a, bbox_b) > 0.5:
                            is_duplicate = True
                            # 保留高置信度
                            if region.get("confidence", 0) > existing.get("confidence", 0):
                                seen.remove(existing)
                                seen.append(region)
                            break
                if not is_duplicate:
                    seen.append(region)
            result.extend(seen)

    return result


def post_process_ocr_results(
    texts: List[str],
    confidences: List[float],
    regions: List[Dict[str, Any]]
) -> Tuple[str, float, List[Dict[str, Any]]]:
    """
    智能OCR后处理管线：
    1. IoU去重 → 2. 阅读顺序排序 → 3. 行合并 → 4. 置信度过滤
    """
    if not texts:
        return "", 0.0, []

    # 1. IoU去重
    regions = _deduplicate_ocr_texts(regions)

    # 2. 按阅读顺序排序
    regions = _sort_boxes_reading_order(regions)

    # 3. 置信度过滤
    filtered_regions = [r for r in regions if r.get("confidence", 0) >= 0.3]

    if not filtered_regions:
        # 如果全部过滤掉，至少保留置信度最高的
        if regions:
            filtered_regions = [max(regions, key=lambda r: r.get("confidence", 0))]

    # 4. 合并成文本
    sorted_texts = [r.get("text", "").strip() for r in filtered_regions if r.get("text", "").strip()]
    full_text = "\n".join(sorted_texts)
    avg_conf = sum(r.get("confidence", 0) for r in filtered_regions) / len(filtered_regions) if filtered_regions else 0.0

    return full_text, avg_conf, filtered_regions


def post_process_ocr_text(text: str) -> str:
    """OCR文本后处理：去重、修复常见错误、校正词汇"""
    if not text:
        return text

    # 数字-字母混淆纠正（包装标签常见）
    digit_corrections = {"O": "0", "o": "0", "l": "1", "I": "1", "S": "5", "B": "8"}

    # 常见OCR错误词汇校正（包装场景）
    word_corrections: Dict[str, str] = {
        "Storrge": "Storage",
        "storrge": "storage",
        "Maufacturer": "Manufacturer",
        "maufacturer": "manufacturer",
        "Mantfacturer": "Manufacturer",
        "Prodct": "Product",
        "podct": "product",
        "Specication": "Specification",
        "Ingedients": "Ingredients",
        "ingedients": "ingredients",
        "Expir": "Expiry",
        "Dtae": "Date",
        "dtae": "date",
        "Contet": "Content",
        "contet": "content",
        "Adress": "Address",
        "adress": "address",
    }

    lines = text.split('\n')
    lines = [line.strip() for line in lines if line.strip()]

    # 去除重复行
    seen: List[str] = []
    unique_lines: List[str] = []
    for line in lines:
        if line not in seen:
            seen.append(line)
            unique_lines.append(line)

    # 修复OCR错误
    processed_lines: List[str] = []
    for line in unique_lines:
        # 1. 词汇级校正（先做，避免影响后续数字替换）
        for wrong, correct in word_corrections.items():
            if wrong in line:
                line = line.replace(wrong, correct)

        # 2. 数字校正（仅对包含数字的行做精细替换）
        if re.search(r'\d{2,}', line):
            for wrong, correct in digit_corrections.items():
                # 数字前的字母混淆（如 O1 → 01）
                line = re.sub(rf'(\d){re.escape(wrong)}', lambda m: m.group(1) + correct, line)
                # 数字后的字母混淆（如 1O → 10）
                line = re.sub(rf'{re.escape(wrong)}(\d)', lambda m: correct + m.group(1), line)

        # 3. 清理多余空白
        line = re.sub(r'\s{2,}', ' ', line)

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
