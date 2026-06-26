#!/usr/bin/env python3
"""VL多模态信息提取节点 - 使用视觉语言模型直接从图片提取结构化信息"""
import os
import json
import logging
import re
from typing import Dict, Any, List
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context
try:
    from coze_coding_dev_sdk import LLMClient
except ImportError:
    LLMClient = None
from langchain_core.messages import SystemMessage, HumanMessage
from graphs.state import VlInfoExtractInput, VlInfoExtractOutput

logger = logging.getLogger(__name__)


def vl_info_extract_node(
    state: VlInfoExtractInput,
    config: RunnableConfig,
    runtime: Runtime[Context],
) -> VlInfoExtractOutput:
    """
    title: VL多模态信息提取
    desc: 使用视觉语言模型直接从图片中提取结构化信息，无需依赖OCR引擎。
    integrations: 多模态大语言模型
    """
    ctx = runtime.context
    
    input_file = state.input_file
    detected_scenario = state.detected_scenario
    url = input_file.url
    
    # 场景对应的提取字段定义
    scenario_fields = {
        "packaging": ["品牌", "品名", "规格", "生产日期", "保质期", "厂家", "批号", "许可证号"],
        "finance_receipt": ["票据类型", "金额", "日期", "收款方", "付款方", "票据编号"],
        "finance_statement": ["账户", "币种", "交易明细", "余额", "日期范围"],
        "pharmaceutical": ["通用名", "规格", "批准文号", "生产企业", "用法用量", "注意事项"],
        "contract": ["合同编号", "甲方", "乙方", "金额", "有效期", "签署日期"],
        "id_card": ["证号", "姓名", "性别", "出生日期", "住址", "有效期"],
        "logistics": ["运单号", "寄件人", "收件人", "品名", "重量", "费用"],
        "general_document": ["标题", "摘要", "关键字", "正文"]
    }
    
    fields_to_extract = scenario_fields.get(detected_scenario, scenario_fields["general_document"])
    
    # 读取配置文件获取模型配置
    cfg_file = os.path.join(os.getenv("COZE_WORKSPACE_PATH"), "config/vl_info_extract_llm_cfg.json")
    
    try:
        with open(cfg_file, 'r', encoding='utf-8') as fd:
            _cfg: Dict[str, Any] = json.load(fd)
    except Exception as e:
        logger.warning(f"无法加载配置文件，使用默认配置: {e}")
        _cfg = {
            "config": {
                "model": "doubao-seed-1-8-251228",
                "temperature": 0.1
            },
            "sp": "你是信息提取专家。",
            "up": "提取图片中的信息"
        }
    
    llm_config = _cfg.get("config", {})
    model_id = llm_config.get("model", "doubao-seed-1-8-251228")
    temperature = llm_config.get("temperature", 0.1)
    sp = _cfg.get("sp", "")
    
    # 构建提取提示词
    extraction_prompt = f"""请从这张图片中提取以下字段的信息：
场景类型: {detected_scenario}
需要提取的字段: {', '.join(fields_to_extract)}

请以JSON格式输出提取结果，格式如下：
{
  "scenario": "{detected_scenario}",
  "extracted_fields": {
    "字段名": "提取的值（如果图片中不存在该字段，请填写空字符串）"
  },
  "confidence": 0.0-1.0的置信度评分,
  "extraction_notes": "提取过程中的备注信息"
}

注意：
1. 只提取图片中实际存在的信息，不要编造或猜测
2. 置信度评分基于图片清晰度和信息完整性
3. 如果某个字段在图片中不存在，请填写空字符串"""
    
    # 构建多模态消息（图片URL直接传递）
    messages = [
        SystemMessage(content=sp),
        HumanMessage(content=[
            {
                "type": "text",
                "text": extraction_prompt
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": url
                }
            }
        ])
    ]
    
    # 使用多模态LLM客户端
    client = LLMClient(ctx=ctx)
    
    try:
        response = client.invoke(
            messages=messages,
            model=model_id,
            temperature=temperature
        )
        
        # 解析响应内容
        content = response.content
        if isinstance(content, str):
            result_text = content.strip()
        elif isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif isinstance(item, str):
                    text_parts.append(item)
            result_text = " ".join(text_parts).strip()
        else:
            result_text = str(content).strip()
        
        # 从响应中提取JSON
        json_pattern = r'\{[^{}]*"scenario"[^{}]*\}'
        matches = re.findall(json_pattern, result_text, re.DOTALL)
        
        if matches:
            json_str = matches[-1]  # 取最后一个匹配
            try:
                structured_data = json.loads(json_str)
            except json.JSONDecodeError:
                # 尝试更宽松的JSON解析
                structured_data = {
                    "scenario": detected_scenario,
                    "extracted_fields": {},
                    "confidence": 0.5,
                    "extraction_notes": "JSON解析失败，使用降级处理"
                }
        else:
            structured_data = {
                "scenario": detected_scenario,
                "extracted_fields": {},
                "confidence": 0.5,
                "extraction_notes": "未找到有效JSON，使用降级处理"
            }
        
        # 计算提取统计信息
        extracted_fields = structured_data.get("extracted_fields", {})
        fields_attempted = len(fields_to_extract)
        fields_extracted = sum(1 for v in extracted_fields.values() if v and v.strip())
        extraction_confidence = structured_data.get("confidence", 0.5)
        
        logger.info(f"VL信息提取完成: scenario={detected_scenario}, fields_extracted={fields_extracted}/{fields_attempted}, confidence={extraction_confidence:.2f}")
        
        return VlInfoExtractOutput(
            structured_data=structured_data,
            extraction_confidence=extraction_confidence,
            fields_attempted=fields_attempted,
            fields_extracted=fields_extracted,
            extraction_method="vl_multimodal"
        )
        
    except Exception as e:
        logger.error(f"VL信息提取失败: {e}")
        # 降级处理：返回空结构
        return VlInfoExtractOutput(
            structured_data={
                "scenario": detected_scenario,
                "extracted_fields": {},
                "confidence": 0.0,
                "extraction_notes": f"VL提取失败: {str(e)[:100]}"
            },
            extraction_confidence=0.0,
            fields_attempted=len(fields_to_extract),
            fields_extracted=0,
            extraction_method="vl_multimodal_fallback"
        )