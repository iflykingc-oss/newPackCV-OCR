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
import hashlib
import json
import requests
from typing import List, Dict, Any, Optional, Tuple
from io import BytesIO
from collections import OrderedDict

import cv2
import numpy as np
from PIL import Image

from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context
from graphs.state import OCRRecognizeInput, OCRRecognizeOutput
from utils.file.file import File

logger = logging.getLogger(__name__)

# ==================== OCR结果缓存 ====================
# LRU缓存：避免重复处理相同图片（增大至200，适合批量场景）
_OCR_CACHE_MAX_SIZE = 200
_ocr_cache: OrderedDict = OrderedDict()

def _cache_key(image_url: str, quality_score: float) -> str:
    """生成缓存key（URL hash + 质量分）"""
    raw = f"{image_url}|{quality_score:.0f}"
    return hashlib.md5(raw.encode()).hexdigest()

def _get_cached(key: str) -> Optional[Tuple[str, float, List[Dict[str, Any]], str]]:
    """从LRU缓存获取"""
    if key in _ocr_cache:
        _ocr_cache.move_to_end(key)
        logger.info(f"OCR缓存命中: {key[:12]}")
        return _ocr_cache[key]
    return None

def _set_cache(key: str, value: Tuple[str, float, List[Dict[str, Any]], str]):
    """写入LRU缓存"""
    _ocr_cache[key] = value
    if len(_ocr_cache) > _OCR_CACHE_MAX_SIZE:
        _ocr_cache.popitem(last=False)

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
_paddle_ocr_instance = None


def _get_paddle_ocr():
    """获取PaddleOCR单例（中文场景最佳引擎）"""
    global _paddle_ocr_instance
    if _paddle_ocr_instance is None:
        try:
            from paddleocr import PaddleOCR
            _paddle_ocr_instance = PaddleOCR(
                use_angle_cls=True, lang='ch',
                use_gpu=False, show_log=False,
                det_db_thresh=0.3, det_db_box_thresh=0.3,
                rec_batch_num=6
            )
            logger.info("PaddleOCR实例初始化成功，专注中文场景文本")
        except Exception as e:
            logger.warning(f"PaddleOCR初始化失败: {e}")
    return _paddle_ocr_instance


def _get_rapid_ocr():
    """获取RapidOCR单例实例（使用优化配置）"""
    global _rapid_ocr_instance
    if _rapid_ocr_instance is None and _RAPID_OCR_AVAILABLE:
        try:
            cfg_path = os.path.join(os.getenv("COZE_WORKSPACE_PATH", ""), "config/rapidocr_optimized.yaml")
            if os.path.exists(cfg_path):
                _rapid_ocr_instance = RapidOCR(config_path=cfg_path)
                logger.info(f"RapidOCR实例初始化成功（使用优化配置）")
            else:
                _rapid_ocr_instance = RapidOCR()
                logger.info("RapidOCR实例初始化成功（默认配置）")
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


# ==================== 布局分析（Direction ①） ====================

def _layout_column_detection(img: np.ndarray, gap_ratio_thresh: float = 0.08) -> List[Dict[str, Any]]:
    """
    基于垂直投影的列布局检测
    返回区域列表：[{"x1": ..., "x2": ..., "y1": ..., "y2": ...}, ...]
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    # 二值化
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 垂直投影：每列的非零像素数
    v_proj = np.sum(binary, axis=0) // 255

    # 找出空白列（近乎无文字的列）
    max_proj = np.max(v_proj) if np.max(v_proj) > 0 else 1
    blank_cols = v_proj < (max_proj * 0.02)

    # 合并相邻空白列 → 空白区域
    gaps: List[Tuple[int, int]] = []
    in_gap = False
    gap_start = 0
    for col in range(w):
        if blank_cols[col] and not in_gap:
            in_gap = True
            gap_start = col
        elif not blank_cols[col] and in_gap:
            in_gap = False
            gap_width = col - gap_start
            if gap_width > w * gap_ratio_thresh:  # 足够的间隔才视为列分割
                gaps.append((gap_start, col))
    if in_gap and (w - gap_start) > w * gap_ratio_thresh:
        gaps.append((gap_start, w))

    if not gaps:
        return [{"x1": 0, "x2": w, "y1": 0, "y2": h}]  # 单列

    # 按空白区域分割为多个列区域
    regions = []
    prev_end = 0
    for gs, ge in gaps:
        if ge - prev_end > 20:  # 忽略太小的区域
            regions.append({"x1": prev_end, "x2": gs, "y1": 0, "y2": h})
        prev_end = ge
    if w - prev_end > 20:
        regions.append({"x1": prev_end, "x2": w, "y1": 0, "y2": h})

    # 如果没有有效区域，回退到整图
    if not regions:
        return [{"x1": 0, "x2": w, "y1": 0, "y2": h}]

    return regions


def _layout_section_detection(img: np.ndarray) -> List[Dict[str, Any]]:
    """
    水平投影段落检测：识别不同内容区域（标题区/成分区/表格式区域）
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 水平投影
    h_proj = np.sum(binary, axis=1) // 255

    # 平滑投影
    kernel_size = max(1, h // 50)
    h_proj_smooth = np.convolve(h_proj, np.ones(kernel_size) / kernel_size, mode='same')

    max_proj = np.max(h_proj_smooth) if np.max(h_proj_smooth) > 0 else 1
    text_rows = h_proj_smooth > (max_proj * 0.05)

    # 合并相邻行 → 段落
    sections: List[Tuple[int, int]] = []
    in_section = False
    sec_start = 0
    for row in range(h):
        if text_rows[row] and not in_section:
            in_section = True
            sec_start = row
        elif not text_rows[row] and in_section:
            in_section = False
            if row - sec_start > 15:  # 忽略太小的段落
                sections.append((sec_start, row))
    if in_section and (h - sec_start) > 15:
        sections.append((sec_start, h))

    # 转换为区域格式
    regions = []
    for ys, ye in sections:
        # 每个段落也考虑列分割
        col_regions = _layout_column_detection(img[ys:ye, :, :] if len(img.shape) == 3 else img[ys:ye, :])
        for cr in col_regions:
            regions.append({"x1": cr["x1"], "x2": cr["x2"], "y1": ys + cr["y1"], "y2": ys + cr.get("y2", ye)})

    if not regions:
        return [{"x1": 0, "x2": w, "y1": 0, "y2": h}]

    return regions


def _layout_aware_ocr(img: np.ndarray) -> Tuple[str, float, List[Dict[str, Any]]]:
    """
    布局感知OCR：先检测版面布局，再分区域OCR，最后合并结果
    解决多列文本混淆、表格混入等问题
    """
    h, w = img.shape[:2]

    # 第一步：列检测（判断是否是分栏布局）
    col_regions = _layout_column_detection(img)

    if len(col_regions) <= 1:
        # 单列 → 进一步做段落检测 + 直接OCR
        sec_regions = _layout_section_detection(img)
        if len(sec_regions) <= 3:  # 段落少，直接整图OCR
            return multi_scale_ocr(img)
        else:
            # 分段落OCR，结果按阅读顺序合并
            all_texts = []
            all_confs = []
            all_regions = []
            for sec in sec_regions:
                sub_img = img[sec["y1"]:sec["y2"], sec["x1"]:sec["x2"]]
                if sub_img.shape[0] < 10 or sub_img.shape[1] < 10:
                    continue
                text, conf, regions = multi_scale_ocr(sub_img)
                if text and conf > 0:
                    # 调整坐标到原图
                    for r in regions:
                        bbox = r.get("bbox", [])
                        if len(bbox) >= 4:
                            r["bbox"] = [bbox[0] + sec["x1"], bbox[1] + sec["y1"],
                                          bbox[2] + sec["x1"], bbox[3] + sec["y1"]]
                    all_texts.append(text)
                    all_confs.append(conf)
                    all_regions.extend(regions)
            merged_text = "\n".join(all_texts) if all_texts else ""
            avg_conf = np.mean(all_confs) if all_confs else 0
            return merged_text, float(avg_conf), all_regions
    else:
        # 多列 → 每列独立OCR → 按阅读顺序合并（先左后右）
        all_texts = []
        all_confs = []
        all_regions = []
        col_regions.sort(key=lambda r: r["x1"])  # 先左后右

        for cr in col_regions:
            sub_img = img[cr["y1"]:cr["y2"], cr["x1"]:cr["x2"]]
            if sub_img.shape[0] < 10 or sub_img.shape[1] < 10:
                continue
            text, conf, regions = multi_scale_ocr(sub_img)
            if text and conf > 0:
                for r in regions:
                    bbox = r.get("bbox", [])
                    if len(bbox) >= 4:
                        r["bbox"] = [bbox[0] + cr["x1"], bbox[1] + cr["y1"],
                                      bbox[2] + cr["x1"], bbox[3] + cr["y1"]]
                all_texts.append(text)
                all_confs.append(conf)
                all_regions.extend(regions)

        merged_text = "\n".join(all_texts) if all_texts else ""
        avg_conf = np.mean(all_confs) if all_confs else 0
        return merged_text, float(avg_conf), all_regions


# ==================== 多尺度OCR检测 ====================

def multi_scale_ocr(img: np.ndarray) -> Tuple[str, float, List[Dict[str, Any]]]:
    """
    多尺度OCR检测：
    1. 原始尺度 1x 检测（通用）
    2. 小图 0.5x 检测（捕获大文字）
    3. 融合去重合并结果
    """
    if not _RAPID_OCR_AVAILABLE:
        return "", 0.0, []

    h, w = img.shape[:2]
    all_texts: List[str] = []
    all_confs: List[float] = []
    all_regions: List[Dict[str, Any]] = []

    # 尺度1: 原始尺度
    text1, conf1, regions1 = ocr_with_rapid(img)
    if text1:
        all_texts.append(text1)
        all_confs.append(conf1)
        all_regions.extend(regions1)
        logger.info(f"多尺度1x: {len(regions1)}行, conf={conf1:.4f}")

    # 尺度2: 0.5x（原始过大时启用，捕获大文字）
    if max(h, w) > 1000:
        try:
            small_h, small_w = h // 2, w // 2
            img_small = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)
            text2, conf2, regions2 = ocr_with_rapid(img_small)
            if text2 and regions2:
                # 坐标还原到原始尺寸
                for r in regions2:
                    bbox = r.get("bbox", [])
                    if len(bbox) == 4:
                        r["bbox"] = [bbox[0] * 2, bbox[1] * 2, bbox[2] * 2, bbox[3] * 2]
                # 对0.5x结果去重（如果与1x高度重叠则跳过）
                merged_regions = _merge_regions(all_regions, regions2, iou_threshold=0.3)
                if len(merged_regions) > len(all_regions):
                    all_regions = merged_regions
                    for r in regions2:
                        if r not in all_regions:
                            all_texts.append(r.get("text", ""))
                            all_confs.append(r.get("confidence", 0.0))
                    logger.info(f"多尺度0.5x: {len(regions2)}行新增, 总计{len(all_regions)}行")
                else:
                    logger.info(f"多尺度0.5x: {len(regions2)}行, 全部重复跳过")
        except Exception as e:
            logger.warning(f"多尺度0.5x失败: {e}")

    # 最终后处理
    if not all_regions:
        return "", 0.0, []

    final_texts = [r["text"] for r in all_regions]
    final_confs = [r["confidence"] for r in all_regions]
    full_text, avg_conf, processed = post_process_ocr_results(final_texts, final_confs, all_regions)

    return full_text, avg_conf, processed


def _merge_regions(
    regions_a: List[Dict[str, Any]],
    regions_b: List[Dict[str, Any]],
    iou_threshold: float = 0.3
) -> List[Dict[str, Any]]:
    """合并两个区域列表，基于IoU去重"""
    merged = list(regions_a)
    for r_b in regions_b:
        is_dup = False
        bbox_b = r_b.get("bbox", [])
        for r_a in regions_a:
            bbox_a = r_a.get("bbox", [])
            if _compute_iou(bbox_a, bbox_b) > iou_threshold:
                is_dup = True
                break
        if not is_dup:
            merged.append(r_b)
    return merged


def _compute_iou(bbox_a: List, bbox_b: List) -> float:
    """计算两个bbox的交并比"""
    if len(bbox_a) != 4 or len(bbox_b) != 4:
        return 0.0
    # [x1, y1, x2, y2]
    x1 = max(bbox_a[0], bbox_b[0])
    y1 = max(bbox_a[1], bbox_b[1])
    x2 = min(bbox_a[2], bbox_b[2])
    y2 = min(bbox_a[3], bbox_b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1])
    area_b = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1])
    if area_a + area_b - inter <= 0:
        return 0.0
    return inter / float(area_a + area_b - inter)


# ==================== PaddleOCR引擎 ====================

def ocr_with_paddle(img: np.ndarray) -> Tuple[str, float, List[Dict[str, Any]]]:
    """使用PaddleOCR进行OCR识别（中文场景最优）"""
    try:
        ocr = _get_paddle_ocr()
        if ocr is None:
            return "", 0.0, []

        # paddleocr接收bgr格式的numpy数组
        result = ocr.ocr(img, cls=True)
        if not result or not result[0]:
            return "", 0.0, []

        texts: List[str] = []
        confidences: List[float] = []
        regions: List[Dict[str, Any]] = []

        for item in result[0]:
            bbox = item[0]  # 4点坐标
            text = str(item[1][0])
            conf = float(item[1][1])

            texts.append(text)
            confidences.append(conf)

            # 4点转轴对齐bbox
            if bbox and len(bbox) >= 4:
                xs = [int(p[0]) for p in bbox]
                ys = [int(p[1]) for p in bbox]
                regions.append({
                    "text": text,
                    "confidence": conf,
                    "bbox": [min(xs), min(ys), max(xs), max(ys)],
                    "type": "text",
                    "engine": "paddleocr"
                })

        if not texts:
            return "", 0.0, []

        raw_text = "\n".join(texts)
        avg_conf = sum(confidences) / len(confidences)
        logger.info(f"PaddleOCR识别: {len(texts)}行, 平均置信度={avg_conf:.4f}")
        return raw_text, avg_conf, regions

    except Exception as e:
        logger.warning(f"PaddleOCR识别失败: {e}")
        return "", 0.0, []


# ==================== 文本相似度（引擎融合用） ====================

def _text_similarity(a: str, b: str) -> float:
    """计算两段文本的字符级相似度（用于多引擎结果匹配）"""
    if not a or not b:
        return 0.0
    a_clean = a.strip().lower()
    b_clean = b.strip().lower()
    if a_clean == b_clean:
        return 1.0
    # 最长公共子序列相似度
    shorter, longer = (a_clean, b_clean) if len(a_clean) <= len(b_clean) else (b_clean, a_clean)
    if not shorter:
        return 0.0
    # 简单字符重叠率
    overlap = sum(1 for c in shorter if c in longer)
    return overlap / len(shorter)


# ==================== 多引擎融合 ====================

def multi_engine_ocr(img: np.ndarray, time_budget: float = 30.0) -> Tuple[str, float, List[Dict[str, Any]], str]:
    """
    三引擎并行融合OCR（方向①：多引擎融合升级版）
    引擎: RapidOCR(通用) + PaddleOCR(中文) + Tesseract(保底)
    融合策略: 置信度加权 + 文本相似度匹配 + 区域合并
    """
    start = time.time()

    # ---- 阶段1: 并行执行所有引擎 ----
    results: List[Tuple[str, float, List[Dict[str, Any]], str]] = []

    # 1. RapidOCR（布局感知版本）
    rapid_text, rapid_conf, rapid_regions = "", 0.0, []
    if _RAPID_OCR_AVAILABLE:
        try:
            text, conf, regions = _layout_aware_ocr(img)
            if text:
                rapid_text, rapid_conf, rapid_regions = text, conf, regions
                results.append((text, conf, regions, "rapidocr"))
                logger.info(f"RapidOCR: {len(text)}字 conf={conf:.3f}")
        except Exception as e:
            logger.warning(f"RapidOCR异常: {e}")

    elapsed = time.time() - start
    if elapsed > time_budget * 0.9:
        logger.info(f"时间预算紧张({elapsed:.1f}s)，仅用RapidOCR结果")
        if rapid_text:
            return rapid_text, rapid_conf, rapid_regions, "rapidocr"
        return "", 0.0, [], "timeout"

    # 2. PaddleOCR（中文场景最优）
    paddle_text, paddle_conf, paddle_regions = "", 0.0, []
    try:
        text, conf, regions = ocr_with_paddle(img)
        if text:
            paddle_text, paddle_conf, paddle_regions = text, conf, regions
            results.append((text, conf, regions, "paddleocr"))
            logger.info(f"PaddleOCR: {len(text)}字 conf={conf:.3f}")
    except Exception as e:
        logger.warning(f"PaddleOCR异常: {e}")

    elapsed = time.time() - start
    if elapsed > time_budget * 0.9:
        if results:
            best = max(results, key=lambda r: r[1] * len(r[0]))
            logger.info(f"时间预算紧张，选择最佳引擎: {best[3]}")
            return best[:3] + (best[3],)
        if rapid_text:
            return rapid_text, rapid_conf, rapid_regions, "rapidocr"
        return "", 0.0, [], "timeout"

    # 3. Tesseract（保底引擎，仅当前两引擎均无结果时快速扫描）
    tesseract_text, tesseract_conf = "", 0.0
    if not any(r[0] for r in results) and _TESSERACT_AVAILABLE:
        try:
            text, conf = ocr_with_tesseract_multi_psm(img, lang="chi_sim+eng")
            if text:
                tesseract_text, tesseract_conf = text, conf
                results.append((text, conf, [], "tesseract"))
                logger.info(f"Tesseract: {len(text)}字 conf={conf:.3f}")
        except Exception as e:
            logger.warning(f"Tesseract异常: {e}")

    # ---- 阶段2: 置信度加权融合 ----
    if not results:
        logger.warning("所有OCR引擎均未识别到文本")
        return "", 0.0, [], "none"

    # 有PaddleOCR结果时，优先用PaddleOCR（中文包装场景最佳）
    if paddle_text and paddle_conf >= 0.4:
        logger.info(f"★ 融合决策: PaddleOCR优先(conf={paddle_conf:.3f}, {len(paddle_text)}字)")
        # 用RapidOCR结果补充PaddleOCR缺失的文字
        if rapid_text and rapid_conf >= 0.5:
            combined_text = paddle_text
            # 若RapidOCR有额外内容，追加
            if len(rapid_text) > len(paddle_text) * 1.3:
                combined_text += "\n" + rapid_text
                logger.info(f"RapidOCR补充文本，合并后长度={len(combined_text)}")
            paddle_regions.append({"text": f"融合来源: PaddleOCR+RapidOCR", "confidence": max(paddle_conf, rapid_conf), "bbox": [], "type": "meta"})
            return combined_text, max(paddle_conf, rapid_conf), paddle_regions, "paddleocr_fused"
        return paddle_text, paddle_conf, paddle_regions, "paddleocr"

    # RapidOCR优先（但PaddleOCR无结果时）
    if rapid_text and rapid_conf >= 0.4:
        logger.info(f"★ 融合决策: RapidOCR优先(conf={rapid_conf:.3f}, {len(rapid_text)}字)")
        return rapid_text, rapid_conf, rapid_regions, "rapidocr"

    # Tesseract保底
    if tesseract_text:
        logger.info(f"★ 融合决策: Tesseract保底(conf={tesseract_conf:.3f})")
        return tesseract_text, tesseract_conf, [], "tesseract"

    # 最终保底：选置信度×长度最高的
    best = max(results, key=lambda r: r[1] * len(r[0]))
    logger.info(f"★ 融合决策: {best[3]}保底(conf={best[1]:.3f})")
    if best[3] == "paddleocr":
        return best[0], best[1], best[2], "paddleocr_fallback"
    return best[0], best[1], best[2], best[3]


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

    # 3. 置信度过滤（宽松阈值，宁可多留不可漏检）
    filtered_regions = [r for r in regions if r.get("confidence", 0) >= 0.15]

    if not filtered_regions:
        # 全部过滤掉时，保留置信度最高的
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
        # English corrections
        "Storrge": "Storage",
        "storrge": "storage",
        "Storge": "Storage",
        "Storege": "Storage",
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
        "YiHaijiaLi": "YiHaiJiaLi",
        # Chinese common OCR corrections
        "已期": "日期",
        "日朋": "日期",
        "月月": "月",
        "曰": "日",
        "末": "未",
        "己": "已",
        "配料表": "配料表",
        "配科": "配料",
        "食月油": "食用油",
        "植物月旨": "植物脂", 
        "添如剂": "添加剂",
        "食品添如剂": "食品添加剂",
        "净含世": "净含量",
        "净含鲎": "净含量",
        "上产曰期": "生产日期",
        "上产厂商": "生产厂商",
        "保质朋": "保质期",
        "保质朞": "保质期",
        "生产厂高": "生产厂商",
        "阴京": "阴凉",
        "阴京干燥": "阴凉干燥",
        "储藏": "储藏",
        "批身": "批次",
        "批兮": "批次",
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
    if state.corrected_image and state.corrected_image.url:
        image_url = state.corrected_image.url
        logger.info(f"使用弯曲校正后图片: {image_url[:60]}")
    elif state.preprocessed_image and state.preprocessed_image.url:
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

    # ====== Smart OCR Engine (自定义模型/LightOnOCR/DeepSeek-OCR) ======
    if state.ocr_engine_type == "smart" or state.ocr_engine_type == "auto":
        try:
            from utils.ocr_engines.smart_router import SmartOCREngine
            engine_cfg_path = os.path.join(
                os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects"),
                "src/config/engine_adapter_cfg.json"
            )
            if os.path.exists(engine_cfg_path):
                with open(engine_cfg_path) as f:
                    engine_cfg = json.load(f)

                # 合并运行时自定义模型配置（来自GraphInput.custom_model_config）
                runtime_custom = state.custom_model_config
                if runtime_custom and isinstance(runtime_custom, dict):
                    runtime_ocr = runtime_custom.get("ocr", [])
                    # GraphInput中的自定义引擎优先级最高，覆盖配置文件
                    if runtime_ocr:
                        existing_custom = engine_cfg.get("ocr_engines", {}).get("custom_engines", [])
                        engine_cfg["ocr_engines"]["custom_engines"] = runtime_ocr + existing_custom
                        logger.info(f"从GraphInput加载了 {len(runtime_ocr)} 个运行时自定义OCR引擎")

                smart_engine = SmartOCREngine(engine_cfg)
                status = smart_engine.get_engine_status()
                # 检查是否有任何可用引擎（自定义引擎 / LightOn / DeepSeek）
                available_engines = [name for name, s in status.items() if s.get("available", False)]
                # 排除fallback（始终可用）
                advanced_available = [e for e in available_engines if e != "fallback"]

                if advanced_available:
                    logger.info(f"SmartOCR引擎可用: {advanced_available}，尝试高级OCR...")
                    result = smart_engine.recognize(image_url)
                    if result.success and result.confidence >= 0.3 and len(result.raw_text.strip()) > 20:
                        elapsed = time.time() - start_time
                        logger.info(f"SmartOCR成功: engine={result.engine_name}, "
                                    f"conf={result.confidence:.2f}, len={len(result.raw_text)}")
                        return OCRRecognizeOutput(
                            raw_text=result.raw_text,
                            ocr_confidence=result.confidence,
                            engine_used=result.engine_name,
                            corrected_result=result.raw_text,
                            processing_time=elapsed,
                            ocr_regions=result.regions
                        )
                    else:
                        logger.info(f"SmartOCR未达要求, 降级到本地OCR")
                else:
                    logger.info(f"无可用高级OCR引擎, 降级到本地OCR")
        except Exception as e:
            logger.warning(f"SmartOCR初始化失败: {e}, 降级到本地OCR")

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

    # ====== 检查缓存 ======
    quality_score = 50.0  # 默认中等
    if state.processing_info:
        quality_score = state.processing_info.get("quality_score", 50.0)
    cache_key_str = _cache_key(image_url, quality_score)
    cached = _get_cached(cache_key_str)
    if cached:
        cached_text, cached_conf, cached_regions, cached_engine = cached
        elapsed = time.time() - start_time
        logger.info(f"缓存命中! 耗时={elapsed:.2f}s (text_len={len(cached_text)}, conf={cached_conf:.3f})")
        return OCRRecognizeOutput(
            raw_text=cached_text,
            ocr_confidence=cached_conf,
            ocr_regions=cached_regions,
            engine_used=f"cache_{cached_engine}",
            processing_time=elapsed
        )

    # ====== 质量分级处理 ======
    # 质量好(score>70) → 单次RapidOCR + 不缩放
    # 质量中(40-70) → 正常缩放 + 多引擎
    # 质量差(<40) → 弱缩放 + 多引擎 + 上采样重试
    h, w = img.shape[:2]
    max_dim = 1500
    min_dim = min(h, w)
    gray_check = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    lap_var = cv2.Laplacian(gray_check, cv2.CV_64F).var()
    is_low_quality = quality_score < 40 or lap_var < 30 or min_dim < 500
    is_high_quality = quality_score >= 70 and lap_var >= 80 and min_dim >= 600

    if is_high_quality:
        # 高质量图：单次OCR，不缩放，跳过重试
        logger.info(f"高质量图: score={quality_score:.0f}, lap_var={lap_var:.1f}, 快速模式")
        final_text, final_conf, ocr_regions, engine = multi_engine_ocr(img)
        ocr_retried = False
    else:
        # 中/低质量图：原有完整逻辑
        if max(h, w) > max_dim and not is_low_quality:
            scale = max_dim / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            logger.info(f"大图缩放: {w}x{h} -> {new_w}x{new_h} (质量正常)")
        elif max(h, w) > max_dim and is_low_quality:
            # 低质量图只缩放到max_dim+50%，保留更多像素
            relaxed_max = int(max_dim * 1.5)
            if max(h, w) > relaxed_max:
                scale = relaxed_max / max(h, w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                logger.info(f"低质量图弱缩放: {w}x{h} -> {new_w}x{new_h}")
            else:
                logger.info(f"低质量图跳过缩放: {w}x{h}, lap_var={lap_var:.1f}")
        elif is_low_quality:
            logger.info(f"图片质量较低，保留原始尺寸: {w}x{h}, lap_var={lap_var:.1f}")
        else:
            logger.info(f"图片质量正常: {w}x{h}, lap_var={lap_var:.1f}")

    # 多引擎融合OCR
    final_text, final_conf, ocr_regions, engine = multi_engine_ocr(img)

    # ====== 如果第一遍OCR为空，自动尝试多种重试策略 ======
    if not final_text and img is not None:
        logger.info("OCR首遍为空，尝试多种重试策略...")
        try:
            gray_retry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            retry_strategies = [
                ("otsu", None),
                ("adaptive_gaussian", None),
                ("clahe_otsu", None),
            ]
            
            for strategy_name, _ in retry_strategies:
                if strategy_name == "otsu":
                    _, binary = cv2.threshold(gray_retry, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                elif strategy_name == "adaptive_gaussian":
                    binary = cv2.adaptiveThreshold(gray_retry, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                    cv2.THRESH_BINARY, 31, 10)
                else:  # clahe_otsu
                    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
                    enhanced = clahe.apply(gray_retry)
                    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                binary_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
                retry_text, retry_conf, retry_regions, retry_engine = multi_engine_ocr(binary_img)
                if retry_text:
                    final_text, final_conf, ocr_regions, engine = retry_text, retry_conf, retry_regions, f"retry_{strategy_name}"
                    logger.info(f"重试成功: strategy={strategy_name}, text_len={len(final_text)}")
                    break
        except Exception as e:
            logger.warning(f"二值化重试失败: {e}")

    # ====== 如果仍然为空，尝试2x上采样+二值化重试 ======
    if not final_text and img is not None:
        logger.info("OCR仍然为空，尝试2x上采样重试...")
        try:
            h_up, w_up = img.shape[:2]
            upscaled = cv2.resize(img, (w_up * 2, h_up * 2), interpolation=cv2.INTER_CUBIC)
            up_text, up_conf, up_regions, up_engine = multi_engine_ocr(upscaled)
            if up_text:
                final_text, final_conf, ocr_regions, engine = up_text, up_conf, up_regions, f"upscale_{up_engine}"
                logger.info(f"上采样重试成功: text_len={len(final_text)}, conf={up_conf:.4f}")
            else:
                # 上采样后做CLAHE+OTSU再OCR
                up_gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(up_gray)
                _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                binary_up = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
                up_text, up_conf, up_regions, up_engine = multi_engine_ocr(binary_up)
                if up_text:
                    final_text, final_conf, ocr_regions, engine = up_text, up_conf, up_regions, "upscale_otsu"
                    logger.info(f"上采样+OTSU重试成功: text_len={len(final_text)}, conf={up_conf:.4f}")
        except Exception as e:
            logger.warning(f"上采样重试失败: {e}")

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

    # 写入缓存（非空结果）
    if final_text and len(final_text) > 5:
        _set_cache(cache_key_str, (final_text, final_conf, ocr_regions, engine))

    return OCRRecognizeOutput(
        raw_text=final_text,
        ocr_confidence=final_conf,
        engine_used=engine,
        processing_time=elapsed_time,
        ocr_regions=ocr_regions
    )
