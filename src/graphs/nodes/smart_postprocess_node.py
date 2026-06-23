# -*- coding: utf-8 -*-
"""
智能后处理节点 (Smart Postprocess Node) - V5.9
合并 knowledge_inference + category_template，一次LLM调用完成知识推理+品类匹配+字段验证
减少串行LLM调用次数，降低端到端延迟
"""

import os
import json
import re
import logging
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context
from coze_coding_dev_sdk import LLMClient
from langchain_core.messages import SystemMessage, HumanMessage
from jinja2 import Template

from graphs.state import SmartPostprocessInput, SmartPostprocessOutput

logger = logging.getLogger(__name__)


# ==================== 品类模板定义（从category_template_node迁移）====================

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
    },
    "个人护理": {
        "required_fields": ["brand", "product_name", "specification", "manufacturer",
                            "ingredients", "usage", "license_number"],
        "optional_fields": ["production_date", "shelf_life", "standard", "features",
                            "storage_condition", "warning", "skin_type", "warnings"],
        "field_priority": {
            "ingredients": 10, "usage": 9, "skin_type": 8,
            "license_number": 8, "specification": 7, "warnings": 6,
        },
        "validation_rules": {
            "license_number": r"国妆特字\w+|卫妆准字\w+",
            "specification": r"\d+\s*(g|kg|ml|L|克|毫升|升)",
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
    },
    "其他": {
        "required_fields": ["brand", "product_name", "specification", "manufacturer"],
        "optional_fields": ["production_date", "shelf_life", "standard", "features",
                            "usage", "barcode", "license_number", "storage_condition",
                            "ingredients"],
        "field_priority": {"specification": 8, "manufacturer": 7},
        "validation_rules": {},
    },
}


def _detect_category(ocr_data: Dict[str, Any], raw_text: str = "") -> str:
    """基于OCR数据和原始文本检测产品品类"""
    pt = ocr_data.get("product_type", "")
    if pt and pt in CATEGORY_TEMPLATES:
        return pt

    category_keywords: Dict[str, List[str]] = {
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
    for field in sorted(priority.keys(), key=lambda f: -priority[f]):
        if field in data:
            sorted_data[field] = data[field]
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
        s = _normalize_value(value)
        if not re.search(pattern, s):
            result["valid"] = False
            result["issues"].append(f"Format does not match expected pattern: {pattern}")
    return result


def _normalize_value(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def _infer_brand_from_rawtext(ocr_data: Dict[str, Any], raw_text: str) -> Dict[str, Any]:
    """从raw_text中补全日化/个人护理品类的品牌和品名"""
    result: Dict[str, Any] = {}
    category = ocr_data.get("product_type", ocr_data.get("detected_category", ""))
    if category not in ("日化", "个人护理", "日化清洁", "个人护理"):
        return result

    lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
    non_noise_lines = [line for line in lines if len(line) >= 2 and not re.match(r'^[\d\s\W]+$', line)]

    if not ocr_data.get("brand"):
        for line in non_noise_lines[:5]:
            if any(kw in line for kw in ["成分", "配料", "用途", "水、", "月桂", "椰油"]):
                continue
            if 2 <= len(line) <= 20 and not re.search(r'[：:、，。；]$', line):
                result["brand"] = line
                break

        first_line = non_noise_lines[0] if non_noise_lines else ""
        if not result.get("brand"):
            if re.match(r'^[水月桂椰油聚二]', first_line):
                result["brand"] = None
                result["brand_unrecognized"] = True

    if not ocr_data.get("product_name"):
        for i, line in enumerate(non_noise_lines):
            if "品名" in line or "名称" in line or "产品" in line:
                match = re.search(r'[：:]\s*(.+)', line)
                if match:
                    result["product_name"] = match.group(1).strip()
                    break
                elif i + 1 < len(non_noise_lines):
                    result["product_name"] = non_noise_lines[i + 1]
                    break

    if not ocr_data.get("specification"):
        for line in lines:
            match = re.search(r'(\d+[\s]*[gkgmlL袋盒瓶包]|\d+[\s]*(克|千克|毫升|升|袋|盒|瓶|包))', line)
            if match:
                result["specification"] = match.group(1)
                break

    return result


def smart_postprocess_node(
    state: SmartPostprocessInput,
    config: RunnableConfig,
    runtime: Runtime[Context]
) -> SmartPostprocessOutput:
    """
    title: 智能后处理
    desc: 合并知识推理+品类模板，一次LLM调用完成推理补全+品类匹配+字段验证+优先级重排，减少串行LLM调用
    integrations: 大语言模型
    """
    ctx = runtime.context

    ocr_data = state.fused_structured_data or {}
    raw_text = state.raw_text or ""

    # ── Step 1: 品类检测（规则匹配，无需LLM）──
    if state.detected_category:
        detected_category = state.detected_category
    else:
        detected_category = _detect_category(ocr_data, raw_text)
    template = CATEGORY_TEMPLATES.get(detected_category, CATEGORY_TEMPLATES["其他"])

    # ── Step 2: 模板校验（规则匹配，无需LLM）──
    required = template["required_fields"]
    optional = template["optional_fields"]
    present_fields = {k: v for k, v in ocr_data.items() if v and v != "N/A" and k != "product_type"}
    missing_required = [f for f in required if f not in present_fields]

    # 字段验证
    validation_results: Dict[str, Any] = {}
    for field, value in present_fields.items():
        validation_results[field] = _validate_field(field, value, template)

    # 字段优先级重排
    reordered = _apply_field_priority(ocr_data, template)

    # 覆盖率
    total_required = len(required)
    present_required = sum(1 for f in required if f in present_fields)
    coverage = present_required / total_required if total_required else 1.0

    # 补全建议
    completion_suggestions: List[Dict[str, str]] = []
    for f in missing_required:
        hint = template.get("extraction_hints", {}) if "extraction_hints" in template else {}
        completion_suggestions.append({
            "field": f,
            "priority": "high",
            "hint": hint.get(f, f"Please extract field: {f}")
        })

    # 品牌/品名推断（日化/个人护理）
    if detected_category in ("日化", "个人护理", "日化清洁", "个人护理") and raw_text:
        infer_result = _infer_brand_from_rawtext(reordered, raw_text)
        if infer_result:
            inferred_brand = infer_result.get("brand")
            inferred_product_name = infer_result.get("product_name")
            if inferred_brand and (not reordered.get("brand") or str(reordered.get("brand")) in ("null", "None", "")):
                reordered["brand"] = inferred_brand
            if inferred_product_name and (not reordered.get("product_name") or str(reordered.get("product_name")) in ("null", "None", "")):
                reordered["product_name"] = inferred_product_name

    # ── Step 3: 知识推理LLM（仅当有缺失必填字段时调用）──
    inferred_fields: List[Dict[str, Any]] = []
    inferred_product_type = ""

    if missing_required:
        try:
            cfg_file = os.path.join(os.getenv("COZE_WORKSPACE_PATH", ""), config['metadata']['llm_cfg'])
            with open(cfg_file, 'r', encoding='utf-8') as f:
                _cfg = json.load(f)

            llm_config = _cfg.get("config", {})
            sp = _cfg.get("sp", "")
            up_template = _cfg.get("up", "")

            # 注入品类模板信息到SP中，提升推理精准度
            category_context = f"\n\n## 当前品类：{detected_category}\n"
            category_context += f"必填字段：{json.dumps(required, ensure_ascii=False)}\n"
            category_context += f"缺失字段：{json.dumps(missing_required, ensure_ascii=False)}\n"
            category_context += f"验证规则：{json.dumps(template.get('validation_rules', {}), ensure_ascii=False)}\n"

            enhanced_sp = sp + category_context
            up = Template(up_template).render(
                extracted_data=json.dumps(reordered, ensure_ascii=False, indent=2),
                raw_text=raw_text,
                product_type=detected_category
            )

            client = LLMClient(ctx=ctx)
            messages = [
                SystemMessage(content=enhanced_sp),
                HumanMessage(content=up)
            ]

            response = client.invoke(
                messages=messages,
                model=llm_config.get("model", "doubao-seed-2-0-pro-260215"),
                temperature=llm_config.get("temperature", 0.1),
                top_p=llm_config.get("top_p", 0.95),
                max_tokens=llm_config.get("max_completion_tokens", 2000),
            )

            result_text = ""
            if isinstance(response.content, str):
                result_text = response.content.strip()
            elif isinstance(response.content, list):
                text_parts: List[str] = []
                for item in response.content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    elif isinstance(item, str):
                        text_parts.append(item)
                result_text = " ".join(text_parts).strip()
            else:
                result_text = str(response.content)

            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                try:
                    parsed = json.loads(json_str)
                    inferred_fields = parsed.get("inferred_fields", [])
                    inferred_product_type = parsed.get("product_type", detected_category)
                    # 将推断的字段合并到reordered中
                    for inf in inferred_fields:
                        field_name = inf.get("field", "")
                        field_value = inf.get("value", "")
                        if field_name and field_value and field_name not in present_fields:
                            reordered[field_name] = field_value
                except json.JSONDecodeError:
                    logger.warning("知识推理结果JSON解析失败")

        except Exception as e:
            logger.error(f"知识推理调用失败: {str(e)}")
            inferred_product_type = detected_category

    logger.info(
        f"[智能后处理] 品类={detected_category}，"
        f"必填{present_required}/{total_required}（覆盖率{coverage:.0%}），"
        f"推理补全{len(inferred_fields)}个字段"
    )

    return SmartPostprocessOutput(
        detected_category=detected_category,
        template_name=detected_category,
        required_fields=required,
        optional_fields=optional,
        missing_required_fields=missing_required,
        field_coverage=round(coverage, 4),
        field_validation=validation_results,
        completion_suggestions=completion_suggestions,
        reordered_data=reordered,
        inferred_fields=inferred_fields,
        inferred_product_type=inferred_product_type or detected_category,
    )
