"""OCR文本后处理：条码过滤 + 中文纠错 + 营养成分表行列重排

独立模块，无重量级依赖，可被任何脚本直接 import。
"""

import os
import re
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


# ── 条码噪声过滤 ─────────────────────────────────────────────────

BARCODE_PATTERNS = [
    re.compile(r'^[\d\s\-]{13,}$'),
    re.compile(r'^\d{13}\s*$'),
    re.compile(r'^\d{12}\s*$'),
    re.compile(r'^[><]*\d{13,}[><]*$'),
]


# ── OCR纠错词典 ──────────────────────────────────────────────────

WORD_CORRECTIONS: Dict[str, str] = {
    # English corrections
    "Storrge": "Storage", "storrge": "storage", "Storge": "Storage",
    "Storege": "Storage", "Maufacturer": "Manufacturer", "maufacturer": "manufacturer",
    "Mantfacturer": "Manufacturer", "Prodct": "Product", "podct": "product",
    "Specication": "Specification", "Ingedients": "Ingredients", "ingedients": "ingredients",
    "Expir": "Expiry", "Dtae": "Date", "dtae": "date", "Contet": "Content",
    "contet": "content", "Adress": "Address", "adress": "address",
    "Protin": "Protein", "protin": "protein",
    "Carbohydate": "Carbohydrate", "carbohydate": "carbohydrate",
    "Saturaed": "Saturated", "saturaed": "saturated",
    "Chlesterol": "Cholesterol", "chlesterol": "cholesterol",
    # 营养成分表英文常见OCR错误（仅匹配独立单词，避免误替换）
    # 这些规则在词汇级校正阶段使用 word boundary 检查
    # 中文基础
    "已期": "日期", "日朋": "日期", "月月": "月", "曰": "日",
    "末": "未", "己": "已", "配科": "配料",
    "食月油": "食用油", "植物月旨": "植物脂",
    "添如剂": "添加剂", "食品添如剂": "食品添加剂",
    "净含世": "净含量", "净含鲎": "净含量",
    "上产曰期": "生产日期", "上产厂商": "生产厂商",
    "保质朋": "保质期", "保质朞": "保质期",
    "生产厂高": "生产厂商", "阴京": "阴凉", "阴京干燥": "阴凉干燥",
    "批身": "批次", "批兮": "批次",
    # 营养成分表 — RapidOCR高频错别字
    "膳足": "膳食", "膳含": "膳食", "膳石": "膳食",
    "碳水化台物": "碳水化合物", "碳水化合 物": "碳水化合物",
    "碳水名古合物": "碳水化合物", "碳水名合物": "碳水化合物",
    "蛋自质": "蛋白质", "蛋自": "蛋白质",
    "月旨肪": "脂肪",
    "膳食红维": "膳食纤维",
    "维生紊": "维生素",
    "营养成分表": "营养成分表",
    "NeVlls": "Sugars",
}

# 需要删除的噪声碎片（正则匹配整行）
NOISE_LINE_PATTERNS = [
    re.compile(r'^限日$'),
    re.compile(r'^力力$'),
]


def post_process_ocr_text(text: str) -> str:
    """OCR文本后处理：过滤条码噪声 + 修复常见错误"""
    if not text:
        return text

    digit_corrections = {"O": "0", "o": "0", "l": "1", "I": "1", "S": "5", "B": "8"}

    # 需要word-boundary匹配的纠错（避免 Calori→Calories 误匹配 Calories）
    boundary_corrections = [
        (re.compile(r'\bCalori\b', re.IGNORECASE), lambda m: "Calories" if m.group()[0].isupper() else "calories"),
        (re.compile(r'\bDietar\b', re.IGNORECASE), lambda m: "Dietary" if m.group()[0].isupper() else "dietary"),
        (re.compile(r'\bFibe\b', re.IGNORECASE), lambda m: "Fiber" if m.group()[0].isupper() else "fiber"),
    ]

    lines = text.split('\n')
    lines = [line.strip() for line in lines if line.strip()]

    # 去除重复行
    seen: List[str] = []
    unique_lines: List[str] = []
    for line in lines:
        if line not in seen:
            seen.append(line)
            unique_lines.append(line)

    processed_lines: List[str] = []
    for line in unique_lines:
        # 0. 过滤条码噪声（纯数字长串）
        if any(p.match(line) for p in BARCODE_PATTERNS):
            continue
        # 过滤噪声碎片
        if any(p.match(line) for p in NOISE_LINE_PATTERNS):
            continue
        # 过滤过短的无意义碎片（1-2个字符且非中文）
        if len(line) <= 2 and not any('\u4e00' <= c <= '\u9fff' for c in line):
            continue

        # 1. 词汇级校正
        for wrong, correct in WORD_CORRECTIONS.items():
            if wrong in line:
                line = line.replace(wrong, correct)

        # 1.5. 边界匹配校正（避免子串误替换）
        for pattern, replacer in boundary_corrections:
            line = pattern.sub(replacer, line)

        # 2. 数字校正
        if re.search(r'\d{2,}', line):
            for wrong, correct in digit_corrections.items():
                line = re.sub(rf'(\d){re.escape(wrong)}', lambda m: m.group(1) + correct, line)
                line = re.sub(rf'{re.escape(wrong)}(\d)', lambda m: correct + m.group(1), line)

        # 3. 清理多余空白
        line = re.sub(r'\s{2,}', ' ', line)

        if line.strip():
            processed_lines.append(line)

    return "\n".join(processed_lines)


def post_process_nutrition_table(text: str, image_path: str = None, image_array=None) -> str:
    """营养成分表处理：优先用 OpenCV 表格检测，fallback 到文本启发式重排

    Args:
        text: OCR识别的文本
        image_path: 图片文件路径（可选）
        image_array: 图片numpy数组（可选，优先级高于image_path）
    """
    if not text:
        return text

    # 优先尝试 CV 表格检测（更精确）
    image_input = image_array if image_array is not None else image_path
    if image_input is not None:
        try:
            from utils.table_detector import extract_nutrition_table
            table_text = extract_nutrition_table(image_input)
            if table_text and len(table_text.strip()) > 20:
                return _replace_nutrition_section(text, table_text)
        except Exception as e:
            logger.debug(f"CV table detection failed, fallback to text heuristic: {e}")

    # fallback: 文本启发式重排
    return _heuristic_nutrition_rearrange(text)


def _replace_nutrition_section(original_text: str, table_text: str) -> str:
    """用 CV 检测到的表格替换原文中的营养成分表区域"""
    lines = original_text.split('\n')

    # 找到营养成分表的起止位置
    start_idx = -1
    end_idx = len(lines)

    nutrition_start_kw = ['营养成分', '营养信息', 'Nutrition Facts', 'Nutrition Information',
                          'Nutritional', '能量', 'Protein', 'Calories']
    nutrition_end_kw = ['生产', '保质', '储藏', '地址', '电话', '批号',
                        'Manufacturer', 'Address', 'Shelf Life', 'Storage',
                        'Product', 'Net Weight', '净含量']

    for i, line in enumerate(lines):
        if start_idx < 0 and any(kw in line for kw in nutrition_start_kw):
            start_idx = i
        elif start_idx >= 0 and any(kw in line for kw in nutrition_end_kw):
            end_idx = i
            break

    if start_idx < 0:
        return original_text

    # 重组文本
    prefix = lines[:start_idx]
    suffix = lines[end_idx:]
    return "\n".join(prefix + [table_text] + suffix)


def _heuristic_nutrition_rearrange(text: str) -> str:
    """文本启发式营养成分表重排（fallback）"""
    if not text:
        return text

    lines = text.split('\n')

    # 检测营养成分表起始位置
    table_start = -1
    table_keywords = [
        '营养成分', '营养信息', 'Nutrition Facts', 'Nutrition Information',
        'Nutritional', '能量', 'Energy', '蛋白质', 'Protein',
    ]
    for i, line in enumerate(lines):
        if any(kw in line for kw in table_keywords):
            table_start = i
            break

    if table_start < 0:
        return text

    # 收集表格行
    table_lines = []
    non_table_prefix = lines[:table_start]
    i = table_start

    table_end_markers = [
        '生产', '保质', '储藏', '地址', '电话', '批号',
        'Manufacturer', 'Address', 'Shelf Life', 'Storage',
        'Product', 'Net Weight', '净含量',
    ]

    while i < len(lines):
        line = lines[i]
        if any(marker in line for marker in table_end_markers) and table_lines:
            break
        table_lines.append(line)
        i += 1

    non_table_suffix = lines[i:]

    if len(table_lines) < 3:
        return text

    # 分析表格行
    text_rows = []
    number_rows = []

    for line in table_lines:
        if any(kw in line for kw in ['项目', 'Item', '每100', 'Per 100', '%NRV', '% NRV', '---', '===']):
            continue

        stripped = line.strip()
        digit_ratio = sum(1 for c in stripped if c.isdigit()) / max(len(stripped), 1)

        if digit_ratio > 0.5:
            number_rows.append(stripped)
        else:
            text_rows.append(stripped)

    if text_rows and number_rows:
        field_pattern = re.compile(
            r'(能量|蛋白质|脂肪|碳水化合物|钠|膳食纤维|'
            r'Calories?|Protein|Fat|Carbohydrate|Sodium|Fiber|'
            r'Sugar|Sugars|Energy|Salt|Cholesterol|'
            r'饱和脂肪|反式脂肪|不饱和脂肪|维生素[ABCD]|钙|铁|锌)',
            re.IGNORECASE
        )

        # 统计文字行中的字段总数和数字行中的数值总数
        total_fields = sum(len(field_pattern.findall(r)) for r in text_rows)
        total_values = sum(len(re.split(r'[\s\t]+', r.strip())) for r in number_rows)

        # 只有当字段数和数值数大致匹配时才重排（避免错配）
        if abs(total_fields - total_values) <= max(total_fields, total_values) * 0.3:
            rearranged = []
            for text_row in text_rows:
                fields = field_pattern.findall(text_row)
                if not fields:
                    rearranged.append(text_row)
                    continue

                if number_rows:
                    nums = number_rows.pop(0)
                    values = re.split(r'[\s\t]+', nums)
                    values = [v for v in values if v]

                    # 仅当数值数与字段数匹配时才配对
                    if len(values) == len(fields):
                        pairs = []
                        for j, field in enumerate(fields):
                            pairs.append(f"{field}: {values[j]}")
                        rearranged.append(" | ".join(pairs))
                    elif len(values) > len(fields):
                        # 数值多于字段，取前N个配对
                        pairs = []
                        for j, field in enumerate(fields):
                            pairs.append(f"{field}: {values[j]}")
                        rearranged.append(" | ".join(pairs))
                        # 剩余数值单独一行
                        remaining = " ".join(values[len(fields):])
                        if remaining.strip():
                            rearranged.append(remaining)
                    else:
                        rearranged.append(text_row)
                        number_rows.insert(0, nums)  # 放回去
                else:
                    rearranged.append(text_row)

            for nums in number_rows:
                rearranged.append(nums)

            table_text = "\n".join(rearranged)
        else:
            table_text = "\n".join(table_lines)
    else:
        table_text = "\n".join(table_lines)

    result_parts = non_table_prefix + [table_text] + non_table_suffix
    return "\n".join(result_parts)


# ==================== VLM 辅助识别 ====================

def vlm_assisted_recognition(image_array, ocr_text: str, ocr_confidence: float,
                              confidence_threshold: float = 0.75,
                              min_text_length: int = 10) -> str:
    """VLM 辅助识别：当 OCR 置信度低或文本过短时，用 VL 模型重新识别

    触发条件：
    1. OCR 置信度 < confidence_threshold
    2. OCR 文本长度 < min_text_length（可能漏识别）
    3. OCR 文本包含大量乱码（中文字符占比异常低）

    策略：
    - 不替换 OCR 成功的结果，只在 OCR 失败/低质量时补充
    - VLM 结果作为辅助参考，与 OCR 结果融合
    """
    if image_array is None:
        return ocr_text

    # 判断是否需要触发 VLM 辅助
    need_vlm = False
    reason = ""

    if ocr_confidence < confidence_threshold:
        need_vlm = True
        reason = f"low_confidence({ocr_confidence:.2f})"
    elif len(ocr_text.strip()) < min_text_length:
        need_vlm = True
        reason = f"short_text({len(ocr_text.strip())}chars)"
    else:
        # 检查中文乱码：计算中文字符占比
        chinese_chars = sum(1 for c in ocr_text if '\u4e00' <= c <= '\u9fff')
        total_chars = len(ocr_text.strip())
        if total_chars > 20 and chinese_chars / total_chars < 0.1 and any('\u4e00' <= c <= '\u9fff' for c in ocr_text):
            # 有中文但占比极低 → 可能是乱码
            need_vlm = True
            reason = f"garbled_chinese({chinese_chars}/{total_chars})"

    if not need_vlm:
        return ocr_text

    logger.info(f"  VLM辅助触发: {reason}")

    try:
        import base64, json
        import requests as req

        # 编码图片
        import cv2
        _, buf = cv2.imencode('.jpg', image_array, [cv2.IMWRITE_JPEG_QUALITY, 90])
        b64 = base64.b64encode(buf).decode('utf-8')
        data_uri = f"data:image/jpeg;base64,{b64}"

        # 尝试多个 VLM 端点
        vlm_endpoints = [
            os.getenv("VL_ENDPOINT", ""),
            os.getenv("OPENAI_API_BASE", ""),
        ]

        vlm_prompt = (
            "Please extract ALL text from this image accurately. "
            "For Chinese text, ensure characters are correctly recognized. "
            "Output the text exactly as it appears, preserving line breaks. "
            "Do not add any explanation."
        )

        for endpoint in vlm_endpoints:
            if not endpoint:
                continue
            try:
                api_key = os.getenv("VL_API_KEY", os.getenv("OPENAI_API_KEY", ""))
                headers = {"Content-Type": "application/json"}
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"

                payload = {
                    "model": os.getenv("VL_MODEL", "gpt-4o-mini"),
                    "messages": [{"role": "user", "content": [
                        {"type": "text", "text": vlm_prompt},
                        {"type": "image_url", "image_url": {"url": data_uri}}
                    ]}],
                    "max_tokens": 4096,
                    "temperature": 0.0,
                }

                resp = req.post(f"{endpoint.rstrip('/')}/v1/chat/completions",
                               json=payload, headers=headers, timeout=30)
                resp.raise_for_status()
                vlm_text = resp.json()["choices"][0]["message"]["content"].strip()

                if vlm_text and len(vlm_text) > len(ocr_text):
                    logger.info(f"  VLM辅助成功: {len(ocr_text)}→{len(vlm_text)} chars")
                    return vlm_text
                else:
                    logger.info(f"  VLM结果未提升: {len(vlm_text)} vs {len(ocr_text)}")
                    return ocr_text

            except Exception as e:
                logger.debug(f"  VLM endpoint {endpoint} failed: {e}")
                continue

        # 所有端点都失败，返回原 OCR 结果
        return ocr_text

    except Exception as e:
        logger.warning(f"  VLM辅助异常: {e}")
        return ocr_text
