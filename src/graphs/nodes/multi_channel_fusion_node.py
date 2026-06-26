# -*- coding: utf-8 -*-
"""
多通道融合节点 (Multi-Channel Fusion Node) - V5.3
对OCR传统管线和VL多模态的结果做字段级加权融合
- 字段级置信度计算
- 跨通道冲突解决
- 互信息校验
"""

import os
import json
import logging
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context
from graphs.state import MultiChannelFusionInput, MultiChannelFusionOutput

logger = logging.getLogger(__name__)


# 字段权重配置：不同字段在不同通道下的可靠性
FIELD_CHANNEL_WEIGHTS: Dict[str, Dict[str, float]] = {
    # OCR通道擅长结构化文本；VL通道擅长视觉理解
    "brand": {"ocr": 0.5, "vl": 0.5},
    "product_name": {"ocr": 0.6, "vl": 0.4},
    "specification": {"ocr": 0.7, "vl": 0.3},
    "production_date": {"ocr": 0.85, "vl": 0.15},
    "shelf_life": {"ocr": 0.85, "vl": 0.15},
    "expiry_date": {"ocr": 0.85, "vl": 0.15},
    "batch_number": {"ocr": 0.9, "vl": 0.1},
    "license_number": {"ocr": 0.9, "vl": 0.1},
    "ingredients": {"ocr": 0.7, "vl": 0.3},
    "nutrition_facts": {"ocr": 0.6, "vl": 0.4},
    "manufacturer": {"ocr": 0.6, "vl": 0.4},
    "manufacturer_address": {"ocr": 0.5, "vl": 0.5},
    "standard": {"ocr": 0.85, "vl": 0.15},
    "storage_condition": {"ocr": 0.7, "vl": 0.3},
    "usage": {"ocr": 0.5, "vl": 0.5},
    "features": {"ocr": 0.3, "vl": 0.7},
    "allergen": {"ocr": 0.6, "vl": 0.4},
    "barcode": {"ocr": 0.95, "vl": 0.05},
    "color": {"ocr": 0.1, "vl": 0.9},
    "shape": {"ocr": 0.1, "vl": 0.9},
    "package_type": {"ocr": 0.4, "vl": 0.6},
}


def _normalize_value(v: Any) -> str:
    """统一格式化为字符串"""
    if v is None:
        return ""
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False, sort_keys=True)
    return str(v).strip()


def _calculate_string_similarity(s1: str, s2: str) -> float:
    """简单字符串相似度（基于最长公共子序列和字符重叠）"""
    if not s1 or not s2:
        return 0.0
    s1_lower = s1.lower()
    s2_lower = s2.lower()
    if s1_lower == s2_lower:
        return 1.0
    # 字符集重叠度
    set1 = set(s1_lower)
    set2 = set(s2_lower)
    if not set1 or not set2:
        return 0.0
    intersection = set1 & set2
    union = set1 | set2
    jaccard = len(intersection) / len(union) if union else 0.0
    # 长度相似度
    len_ratio = min(len(s1), len(s2)) / max(len(s1), len(s2)) if max(len(s1), len(s2)) else 0.0
    return 0.6 * jaccard + 0.4 * len_ratio


def _field_consistency_check(field_name: str, ocr_value: str, vl_value: str) -> float:
    """字段一致性检查：OCR和VL两个通道对同一字段的结果是否一致"""
    return _calculate_string_similarity(ocr_value, vl_value)


def multi_channel_fusion_node(
    state: MultiChannelFusionInput,
    config: RunnableConfig,
    runtime: Runtime[Context]
) -> MultiChannelFusionOutput:
    """
    title: 多通道融合
    desc: 对OCR传统管线和VL多模态的提取结果进行字段级加权融合，解决通道间冲突，提升整体提取精度
    integrations: 大语言模型
    """
    ctx = runtime.context

    ocr_data = state.structured_data or {}
    vl_data = state.vl_extracted_data or {}
    ocr_conf = state.ocr_confidence or 0.7
    vl_conf = state.vl_confidence or 0.7

    # 收集所有字段
    all_fields: set = set()
    all_fields.update(ocr_data.keys())
    all_fields.update(vl_data.keys())
    # 排除元数据字段
    all_fields -= {"product_type", "confidence", "engine", "raw_text", "_source"}

    fused_data: Dict[str, Any] = {}
    field_decisions: List[Dict[str, Any]] = []

    for field in all_fields:
        ocr_val = ocr_data.get(field)
        vl_val = vl_data.get(field)
        ocr_str = _normalize_value(ocr_val)
        vl_str = _normalize_value(vl_val)

        # 通道权重
        weights = FIELD_CHANNEL_WEIGHTS.get(field, {"ocr": 0.5, "vl": 0.5})

        # 通道综合得分 = 通道权重 × 通道置信度
        ocr_score = weights["ocr"] * ocr_conf if ocr_str else 0.0
        vl_score = weights["vl"] * vl_conf if vl_str else 0.0

        # 一致性检查
        if ocr_str and vl_str:
            consistency = _field_consistency_check(field, ocr_str, vl_str)
            if consistency > 0.7:
                # 高一致：优先选择OCR（结构化更强）
                fused_data[field] = ocr_val if ocr_val is not None else vl_val
                decision = "consensus_ocr"
                chosen = "ocr"
            elif consistency > 0.3:
                # 中等一致：选分高者
                if ocr_score >= vl_score:
                    fused_data[field] = ocr_val
                    chosen = "ocr"
                else:
                    fused_data[field] = vl_val
                    chosen = "vl"
                decision = "weighted_choice"
            else:
                # 低一致/冲突：选置信度高者，但标注冲突
                if ocr_score > vl_score * 1.2:
                    fused_data[field] = ocr_val
                    chosen = "ocr"
                elif vl_score > ocr_score * 1.2:
                    fused_data[field] = vl_val
                    chosen = "vl"
                else:
                    # 不分胜负：融合两个值
                    fused_data[field] = f"{ocr_str} | {vl_str}"
                    chosen = "both"
                decision = "conflict_resolution"

            field_decisions.append({
                "field": field,
                "decision": decision,
                "chosen": chosen,
                "ocr_value": ocr_str[:50] if ocr_str else None,
                "vl_value": vl_str[:50] if vl_str else None,
                "ocr_score": round(ocr_score, 3),
                "vl_score": round(vl_score, 3),
                "consistency": round(consistency, 3),
            })
        elif ocr_str:
            # 仅OCR有结果
            fused_data[field] = ocr_val
            field_decisions.append({
                "field": field,
                "decision": "ocr_only",
                "chosen": "ocr",
                "ocr_value": ocr_str[:50],
                "ocr_score": round(ocr_score, 3),
            })
        elif vl_str:
            # 仅VL有结果
            fused_data[field] = vl_val
            field_decisions.append({
                "field": field,
                "decision": "vl_only",
                "chosen": "vl",
                "vl_value": vl_str[:50],
                "vl_score": round(vl_score, 3),
            })
        # 否则两通道都没有

    # 产品类型融合
    if "product_type" not in fused_data:
        ocr_type = ocr_data.get("product_type", "")
        vl_type = vl_data.get("product_type", "")
        if ocr_type and vl_type:
            fused_data["product_type"] = ocr_type if ocr_type == vl_type else vl_type
        elif ocr_type:
            fused_data["product_type"] = ocr_type
        elif vl_type:
            fused_data["product_type"] = vl_type

    # 计算融合置信度
    if field_decisions:
        avg_consistency = sum(d.get("consistency", 0.5) for d in field_decisions) / len(field_decisions)
        # 融合置信度 = (ocr_conf + vl_conf) / 2 * 一致性加成
        fused_confidence = (ocr_conf + vl_conf) / 2 * (0.7 + 0.3 * avg_consistency)
    else:
        fused_confidence = max(ocr_conf, vl_conf)

    # 统计
    consensus_count = sum(1 for d in field_decisions if d["decision"] == "consensus_ocr")
    conflict_count = sum(1 for d in field_decisions if d["decision"] == "conflict_resolution")

    logger.info(
        f"[多通道融合] 完成：共{len(field_decisions)}字段，"
        f"一致{consensus_count}，冲突{conflict_count}，融合置信度={fused_confidence:.3f}"
    )

    return MultiChannelFusionOutput(
        fused_structured_data=fused_data,
        fused_confidence=round(fused_confidence, 4),
        fusion_field_count=len(field_decisions),
        fusion_consensus_count=consensus_count,
        fusion_conflict_count=conflict_count,
        fusion_decisions=field_decisions[:20],  # 限制大小
        fusion_method="weighted_score"
    )
