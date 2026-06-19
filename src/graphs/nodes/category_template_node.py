# -*- coding: utf-8 -*-
"""
品类模板库节点 (Category Template Node) - V5.3
根据产品品类加载不同的字段模板+专有规则
支持品类：食品/饮料/日化/药品/酒类/电子/个人护理/化妆品
"""

import os
import json
import logging
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context
from graphs.state import CategoryTemplateInput, CategoryTemplateOutput
import re

logger = logging.getLogger(__name__)


# ==================== 品类模板定义 ====================

CATEGORY_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "食品": {
        "required_fields": ["brand", "product_name", "specification", "production_date",
                            "shelf_life", "ingredients", "nutrition_facts", "manufacturer",
                            "license_number", "storage_condition"],
        "optional_fields": ["batch_number", "standard", "usage", "features",
                            "allergen", "barcode", "origin", "cooking_method"],
        "field_priority": {
            "production_date": 10, "shelf_life": 10, "ingredients": 9,
            "nutrition_facts": 9, "license_number": 8, "specification": 8,
        },
        "validation_rules": {
            "production_date": r"\d{4}[-/年.]?\d{1,2}[-/月.]?\d{1,2}",
            "shelf_life": r"\d+\s*(天|日|月|年|hours?|days?|months?|years?)",
            "specification": r"\d+\s*(g|kg|ml|L|克|千克|毫升|升|袋|盒|瓶|包)",
            "license_number": r"SC\d+|QS\d{12,}",
        },
        "extraction_hints": {
            "nutrition_facts": "营养成分表，包含能量、蛋白质、脂肪、碳水化合物、钠等",
            "ingredients": "配料表，按含量从高到低排列",
        },
    },
    "饮料": {
        "required_fields": ["brand", "product_name", "specification", "production_date",
                            "shelf_life", "ingredients", "manufacturer", "license_number"],
        "optional_fields": ["batch_number", "standard", "storage_condition", "features",
                            "barcode", "origin", "juice_content"],
        "field_priority": {
            "production_date": 10, "shelf_life": 10, "ingredients": 9,
            "juice_content": 8, "license_number": 8, "specification": 8,
        },
        "validation_rules": {
            "production_date": r"\d{4}[-/年.]?\d{1,2}[-/月.]?\d{1,2}",
            "shelf_life": r"\d+\s*(天|日|月|年)",
            "specification": r"\d+\s*(ml|L|毫升|升|瓶|罐|盒)",
            "license_number": r"SC\d+",
        },
        "extraction_hints": {
            "juice_content": "果汁含量百分比（如果适用）",
        },
    },
    "日化": {
        "required_fields": ["brand", "product_name", "specification", "manufacturer",
                            "license_number", "usage", "ingredients"],
        "optional_fields": ["production_date", "shelf_life", "standard", "features",
                            "storage_condition", "barcode", "warning"],
        "field_priority": {
            "ingredients": 10, "usage": 9, "license_number": 8,
            "specification": 8, "warning": 7,
        },
        "validation_rules": {
            "license_number": r"(?:XK\d{12,})|(?:卫妆准字\w+)|(?:卫消证字\w+)",
            "specification": r"\d+\s*(g|kg|ml|L|克|千克|毫升|升|瓶|袋|盒)",
        },
        "extraction_hints": {
            "ingredients": "化学成分/表面活性剂/功效成分",
            "usage": "使用方法、适用场景",
            "warnings": "注意事项、警示语、禁忌内容",
        },
    },
    "个人护理": {
        "required_fields": ["brand", "product_name", "specification", "manufacturer",
                            "ingredients", "usage", "license_number"],
        "optional_fields": ["production_date", "shelf_life", "standard", "features",
                            "storage_condition", "warning", "skin_type", "warnings"],
        "field_priority": {
            "ingredients": 10, "usage": 9, "skin_type": 8,
            "license_number": 8, "specification": 7,
            "warnings": 6,
        },
        "validation_rules": {
            "license_number": r"国妆特字\w+|卫妆准字\w+",
            "specification": r"\d+\s*(g|kg|ml|L|克|毫升|升)",
        },
        "extraction_hints": {
            "skin_type": "适用肤质（干性/油性/混合性/敏感性）",
            "warning": "使用注意事项、过敏警告",
        },
    },
    "药品": {
        "required_fields": ["brand", "product_name", "specification", "manufacturer",
                            "license_number", "usage", "ingredients", "shelf_life",
                            "production_date", "batch_number", "standard"],
        "optional_fields": ["storage_condition", "features", "warning", "barcode", "dosage"],
        "field_priority": {
            "license_number": 10, "ingredients": 10, "usage": 9,
            "dosage": 9, "batch_number": 8, "shelf_life": 8,
        },
        "validation_rules": {
            "license_number": r"国药准字\w+",
            "specification": r"\d+\s*(mg|g|kg|ml|片|粒|丸|袋|瓶|盒)",
            "shelf_life": r"\d+\s*(月|年|天)",
        },
        "extraction_hints": {
            "ingredients": "主要成分、化学名",
            "usage": "用法用量、适应症",
            "warning": "禁忌、不良反应",
        },
    },
    "酒类": {
        "required_fields": ["brand", "product_name", "specification", "manufacturer",
                            "production_date", "shelf_life", "license_number", "alcohol_content"],
        "optional_fields": ["ingredients", "standard", "storage_condition", "features",
                            "barcode", "origin", "raw_materials"],
        "field_priority": {
            "alcohol_content": 10, "production_date": 9, "license_number": 9,
            "specification": 8, "shelf_life": 7,
        },
        "validation_rules": {
            "alcohol_content": r"\d+\.?\d*\s*(%|vol|度|%)",
            "specification": r"\d+\s*(ml|L|毫升|升|瓶)",
            "license_number": r"SC\d+",
        },
        "extraction_hints": {
            "alcohol_content": "酒精度数，如 53%vol",
            "raw_materials": "原料（高粱/小麦/葡萄等）",
        },
    },
    "电子产品": {
        "required_fields": ["brand", "product_name", "specification", "manufacturer",
                            "model", "standard"],
        "optional_fields": ["features", "usage", "barcode", "warranty", "origin",
                            "material", "color"],
        "field_priority": {
            "model": 10, "specification": 9, "standard": 8,
            "warranty": 7, "material": 6,
        },
        "validation_rules": {
            "model": r"[A-Z0-9][A-Z0-9\-]{3,}",
            "specification": r"\d+\s*(mAh|GB|TB|W|V|A|kg|g|mm|cm|寸|英寸)",
        },
        "extraction_hints": {
            "model": "产品型号，如 iPhone 15 Pro",
            "warranty": "保修期/质保期",
        },
    },
    "化妆品": {
        "required_fields": ["brand", "product_name", "specification", "manufacturer",
                            "license_number", "ingredients", "usage"],
        "optional_fields": ["production_date", "shelf_life", "standard", "features",
                            "storage_condition", "warning", "skin_type", "barcode"],
        "field_priority": {
            "ingredients": 10, "license_number": 10, "usage": 9,
            "skin_type": 8, "shelf_life": 7,
        },
        "validation_rules": {
            "license_number": r"国妆特字\w+|国妆网备字\w+|卫妆准字\w+",
            "specification": r"\d+\s*(g|kg|ml|L|克|毫升|升|片|颗)",
        },
        "extraction_hints": {
            "ingredients": "成分表（INCI国际命名）",
            "usage": "使用方法、注意事项",
        },
    },
    "其他": {
        "required_fields": ["brand", "product_name", "specification", "manufacturer"],
        "optional_fields": ["production_date", "shelf_life", "standard", "features",
                            "usage", "barcode", "license_number", "storage_condition",
                            "ingredients"],
        "field_priority": {
            "specification": 8, "manufacturer": 7,
        },
        "validation_rules": {},
        "extraction_hints": {},
    },
}


def _detect_category(ocr_data: Dict[str, Any], raw_text: str = "") -> str:
    """基于OCR数据和原始文本检测产品品类"""
    # 1. 优先使用结构化数据中已有的product_type
    pt = ocr_data.get("product_type", "")
    if pt and pt in CATEGORY_TEMPLATES:
        return pt

    # 2. 关键词匹配
    category_keywords = {
        "药品": ["国药准字", "OTC", "非处方药", "说明书", "适应症", "用法用量", "禁忌", "不良反应"],
        "酒类": ["酒精度", "%vol", "白酒", "葡萄酒", "啤酒", "高粱", "茅台", "五粮液", "酱香"],
        "化妆品": ["国妆特字", "国妆网备字", "卫妆准字", "INCI", "保湿", "美白", "抗皱", "防晒", "肌肤"],
        "个人护理": ["洗发水", "沐浴露", "牙膏", "牙刷", "洗面奶", "护发素"],
        "日化": ["洗衣液", "洗洁精", "消毒", "清洁剂", "卫消证字", "XK"],
        "饮料": ["果汁", "茶饮料", "碳酸", "矿泉水", "纯净水", "咖啡", "奶茶"],
        "电子产品": ["型号", "mAh", "GB", "内存", "充电器", "锂电池", "5G", "WiFi", "蓝牙"],
        "食品": ["营养成分", "配料表", "保质期", "SC", "生产许可证", "食品生产"],
    }

    text_to_check = (raw_text or "").lower() + " " + json.dumps(ocr_data, ensure_ascii=False).lower()
    category_scores: Dict[str, int] = {}
    for cat, keywords in category_keywords.items():
        score = sum(1 for kw in keywords if kw.lower() in text_to_check)
        if score > 0:
            category_scores[cat] = score

    if category_scores:
        return max(category_scores, key=lambda c: category_scores[c])
    return "其他"


def _apply_field_priority(data: Dict[str, Any], template: Dict[str, Any]) -> Dict[str, Any]:
    """根据字段优先级对数据进行加权排序"""
    priority = template.get("field_priority", {})
    sorted_data: Dict[str, Any] = {}
    # 先按优先级排序的字段
    for field in sorted(priority.keys(), key=lambda f: -priority[f]):
        if field in data:
            sorted_data[field] = data[field]
    # 再加其他字段
    for field, val in data.items():
        if field not in sorted_data:
            sorted_data[field] = val
    return sorted_data


def _validate_field(field: str, value: Any, template: Dict[str, Any]) -> Dict[str, Any]:
    """基于品类模板验证字段"""
    result: Dict[str, Any] = {"value": value, "valid": True, "issues": []}
    if value is None or value == "" or value == "N/A":
        return result
    rules = template.get("validation_rules", {})
    pattern = rules.get(field)
    if pattern:
        s = _normalize(value)
        if not re.search(pattern, s):
            result["valid"] = False
            result["issues"].append(f"格式不符合预期模式: {pattern}")
    return result


def _normalize(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def _infer_brand_from_rawtext(
    ocr_data: Dict[str, Any],
    raw_text: str
) -> Dict[str, Any]:
    """从raw_text中补全日化/个人护理品类的品牌和品名"""
    result: Dict[str, Any] = {}
    category = ocr_data.get("product_type", ocr_data.get("detected_category", ""))
    if category not in ("日化", "个人护理", "日化清洁", "个人护理"):
        return result

    # 品牌常见位置特征：raw_text首行通常为品牌名
    lines = [l.strip() for l in raw_text.split("\n") if l.strip()]
    non_noise_lines = [l for l in lines if len(l) >= 2 and not re.match(r'^[\d\s\W]+$', l)]

    # 如果brand缺失，尝试从raw_text首行非噪声行推断
    if not ocr_data.get("brand"):
        for line in non_noise_lines[:5]:
            # 过滤掉已知的成分关键词行
            if any(kw in line for kw in ["成分", "配料", "用途", "水、", "月桂", "椰油"]):
                continue
            if 2 <= len(line) <= 20 and not re.search(r'[：:、，。；]$', line):
                result["brand"] = line
                break

        # 如果是常见化学成分开头（无品牌），标记为"未识别品牌"
        first_line = non_noise_lines[0] if non_noise_lines else ""
        if not result.get("brand"):
            if re.match(r'^[水月桂椰油聚二]', first_line):
                result["brand"] = None
                result["brand_unrecognized"] = True

    # product_name缺失时，从raw_text搜索品名或第二行
    if not ocr_data.get("product_name"):
        for i, line in enumerate(non_noise_lines):
            if "品名" in line or "名称" in line or "产品" in line:
                # 提取冒号后的内容
                m = re.search(r'[：:]\s*(.+)', line)
                if m:
                    result["product_name"] = m.group(1).strip()
                    break
                elif i + 1 < len(non_noise_lines):
                    result["product_name"] = non_noise_lines[i + 1]
                    break

    # specification缺失时，从raw_text搜索净含量
    if not ocr_data.get("specification"):
        for line in lines:
            m = re.search(r'(\d+[\s]*[gkgmlL袋盒瓶包]|\d+[\s]*(克|千克|毫升|升|袋|盒|瓶|包))', line)
            if m:
                result["specification"] = m.group(1)
                break

    return result


def category_template_node(
    state: CategoryTemplateInput,
    config: RunnableConfig,
    runtime: Runtime[Context]
) -> CategoryTemplateOutput:
    """
    title: 品类模板应用
    desc: 根据产品品类加载不同的字段模板+专有规则，提升字段提取的针对性和准确性
    integrations: 大语言模型
    """
    ctx = runtime.context

    ocr_data = state.fused_structured_data or {}
    raw_text = state.raw_text or ""
    # 优先使用上游已检测的品类
    if state.detected_category:
        detected_category = state.detected_category
        template = CATEGORY_TEMPLATES.get(detected_category, CATEGORY_TEMPLATES["其他"])
    else:
        detected_category = state.product_type_hint or _detect_category(ocr_data, raw_text)
        template = CATEGORY_TEMPLATES.get(detected_category, CATEGORY_TEMPLATES["其他"])

    # 2. 应用模板
    required = template["required_fields"]
    optional = template["optional_fields"]
    all_template_fields = set(required + optional)

    # 3. 分离有/无字段
    present_fields = {k: v for k, v in ocr_data.items() if v and v != "N/A" and k != "product_type"}
    missing_required = [f for f in required if f not in present_fields]
    missing_optional = [f for f in optional if f not in present_fields]

    # 4. 验证
    validation_results: Dict[str, Any] = {}
    for field, value in present_fields.items():
        validation_results[field] = _validate_field(field, value, template)

    # 5. 字段重排（按优先级）
    reordered = _apply_field_priority(ocr_data, template)

    # 6. 计算覆盖率
    total_required = len(required)
    present_required = sum(1 for f in required if f in present_fields)
    coverage = present_required / total_required if total_required else 1.0

    # 7. 提取补全建议
    completion_suggestions: List[Dict[str, str]] = []
    for f in missing_required:
        hint = template.get("extraction_hints", {}).get(f, "")
        completion_suggestions.append({
            "field": f,
            "priority": "high",
            "hint": hint or f"请提取{f}字段"
        })
    for f in missing_optional[:3]:
        hint = template.get("extraction_hints", {}).get(f, "")
        completion_suggestions.append({
            "field": f,
            "priority": "medium",
            "hint": hint or f"可选字段{f}"
        })

    # 8. 品牌/品名推断（针对日化/个人护理品类）
    inferred_brand = None
    inferred_product_name = None
    if detected_category in ("日化清洁", "个人护理") and raw_text:
        infer_result = _infer_brand_from_rawtext(reordered, raw_text)
        if infer_result:
            inferred_brand = infer_result.get("brand")
            inferred_product_name = infer_result.get("product_name")
        if inferred_brand and (not reordered.get("brand") or str(reordered.get("brand")) in ("null", "None", "")):
            reordered["brand"] = inferred_brand
            completion_suggestions.insert(0, {
                "field": "brand",
                "priority": "high",
                "hint": f"从包装文本推断品牌为：{inferred_brand}"
            })
        if inferred_product_name and (not reordered.get("product_name") or str(reordered.get("product_name")) in ("null", "None", "")):
            reordered["product_name"] = inferred_product_name
            completion_suggestions.insert(0, {
                "field": "product_name",
                "priority": "high",
                "hint": f"从包装文本推断产品名为：{inferred_product_name}"
            })

    logger.info(
        f"[品类模板] 品类={detected_category}，"
        f"必填{present_required}/{total_required}（覆盖率{coverage:.0%}），"
        f"待补全{len(missing_required)}必填+{len(missing_optional)}可选"
    )

    return CategoryTemplateOutput(
        detected_category=detected_category,
        required_fields=required,
        optional_fields=optional,
        missing_required_fields=missing_required,
        missing_optional_fields=missing_optional,
        field_coverage=round(coverage, 4),
        field_validation=validation_results,
        completion_suggestions=completion_suggestions,
        reordered_data=reordered,
        template_name=detected_category
    )
