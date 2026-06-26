#!/usr/bin/env python3
"""信息提取节点 - 结构化信息提取"""
import os
import json
import re
import logging
from typing import Dict, Any
from jinja2 import Template
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context
from graphs.state import InfoExtractInput, InfoExtractOutput

logger = logging.getLogger(__name__)


# 场景对应的提取字段模板
SCENARIO_FIELDS = {
    "packaging": ["品牌", "品名", "规格", "生产日期", "保质期", "厂家", "批号"],
    "finance_receipt": ["票据类型", "金额", "日期", "收付款方", "发票号"],
    "finance_statement": ["账户", "币种", "交易明细", "余额"],
    "pharmaceutical": ["通用名", "规格", "批准文号", "生产企业", "有效期"],
    "contract": ["合同编号", "甲方", "乙方", "金额", "有效期"],
    "id_card": ["证号", "姓名", "性别", "出生日期", "住址"],
    "logistics": ["运单号", "寄件人", "收件人", "品名", "重量"],
    "general_document": ["标题", "摘要", "关键字", "正文"]
}


def info_extract_node(
    state: InfoExtractInput,
    config: RunnableConfig,
    runtime: Runtime[Context]
) -> InfoExtractOutput:
    """
    title: 结构化信息提取
    desc: 根据场景类型从OCR文本中提取对应字段的结构化信息
    integrations: 大语言模型
    """
    ctx = runtime.context
    
    # 从配置文件读取LLM配置
    cfg_path = config.get("metadata", {}).get("llm_cfg", "")
    if cfg_path:
        full_cfg_path = os.path.join(os.getenv("COZE_WORKSPACE_PATH", ""), cfg_path)
        try:
            with open(full_cfg_path, 'r', encoding='utf-8') as f:
                llm_cfg = json.load(f)
        except Exception as e:
            logger.warning(f"无法读取配置文件 {full_cfg_path}: {e}")
            llm_cfg = {}
    else:
        llm_cfg = {}
    
    ocr_result = state.ocr_result
    detected_scenario = state.detected_scenario
    
    # 获取该场景的提取字段
    fields_to_extract = SCENARIO_FIELDS.get(detected_scenario, SCENARIO_FIELDS["general_document"])
    
    # 结构化数据提取逻辑：
    # 1. 根据场景确定提取字段
    # 2. 使用关键词匹配和正则表达式提取字段值
    # 3. 组装结构化数据
    
    extracted_fields: Dict[str, Any] = {}
    
    # 关键词匹配提取
    for field_name in fields_to_extract:
        extracted_fields[field_name] = ""
    
    # 尝试从OCR文本中提取关键信息
    # 日期格式匹配
    date_pattern = r'(\d{4}[-年]\d{1,2}[-月]\d{1,2}[日]?|\d{1,2}[-/]\d{1,2}[-/]\d{4})'
    dates = re.findall(date_pattern, ocr_result)
    if dates:
        extracted_fields["生产日期"] = dates[0] if "生产日期" in extracted_fields else ""
        extracted_fields["日期"] = dates[0] if "日期" in extracted_fields else ""
    
    # 金额格式匹配
    amount_pattern = r'(\d+[,.]\d{2}|¥?\d+\.?\d*元|人民币\s*\d+\.?\d*元)'
    amounts = re.findall(amount_pattern, ocr_result)
    if amounts:
        extracted_fields["金额"] = amounts[0] if "金额" in extracted_fields else ""
    
    # 编号/证号格式匹配
    id_pattern = r'[A-Za-z0-9]{8,20}|[\u4e00-\u9fa5]+[A-Za-z0-9]{6,}'
    ids = re.findall(id_pattern, ocr_result)
    if ids:
        for field in ["发票号", "合同编号", "运单号", "批准文号", "证号", "批号"]:
            if field in extracted_fields and ids:
                extracted_fields[field] = ids[0]
                break
    
    # 组装最终结构化数据
    structured_data = {
        "scenario": detected_scenario,
        "raw_text_preview": ocr_result[:200] if len(ocr_result) > 200 else ocr_result,
        "extracted_fields": extracted_fields,
        "extraction_metadata": {
            "text_length": len(ocr_result),
            "fields_attempted": len(fields_to_extract),
            "fields_extracted": sum(1 for v in extracted_fields.values() if v)
        }
    }
    
    logger.info(f"信息提取完成: 场景={detected_scenario}, 提取字段数={len(extracted_fields)}")
    
    return InfoExtractOutput(structured_data=structured_data)