#!/usr/bin/env python3
"""VL多模态场景检测节点 - 使用视觉语言模型直接识别图片场景"""
import os
import json
import logging
from typing import Dict, Any
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context
try:
    from coze_coding_dev_sdk import LLMClient
except ImportError:
    LLMClient = None
from langchain_core.messages import SystemMessage, HumanMessage
from graphs.state import VlScenarioDetectInput, VlScenarioDetectOutput

logger = logging.getLogger(__name__)


def vl_scenario_detector_node(
    state: VlScenarioDetectInput,
    config: RunnableConfig,
    runtime: Runtime[Context],
) -> VlScenarioDetectOutput:
    """
    title: VL多模态场景智能检测
    desc: 使用视觉语言模型直接识别图片类型（包装/金融票据/银行流水/医药/合同/证件/物流/通用文档），返回最匹配的场景类型和置信度。
    integrations: 多模态大语言模型
    """
    ctx = runtime.context
    
    input_file = state.input_file
    url = input_file.url
    
    # 读取配置文件获取模型配置
    cfg_file = os.path.join(os.getenv("COZE_WORKSPACE_PATH"), "config/vl_scenario_detect_llm_cfg.json")
    
    try:
        with open(cfg_file, 'r', encoding='utf-8') as fd:
            _cfg: Dict[str, Any] = json.load(fd)
    except Exception as e:
        logger.warning(f"无法加载配置文件，使用默认配置: {e}")
        _cfg = {
            "config": {
                "model": "doubao-seed-1-8-251228",
                "temperature": 0.3
            },
            "sp": "你是场景识别专家。",
            "up": "识别图片类型"
        }
    
    llm_config = _cfg.get("config", {})
    model_id = llm_config.get("model", "doubao-seed-1-8-251228")
    temperature = llm_config.get("temperature", 0.3)
    sp = _cfg.get("sp", "")
    up = _cfg.get("up", "")
    
    # 构建多模态消息（图片URL直接传递）
    messages = [
        SystemMessage(content=sp),
        HumanMessage(content=[
            {
                "type": "text",
                "text": up
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
            # 处理多模态响应（列表格式）
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif isinstance(item, str):
                    text_parts.append(item)
            result_text = " ".join(text_parts).strip()
        else:
            result_text = str(content).strip()
        
        # 从响应中提取场景类型
        # 定义8个场景的关键词
        scenario_keywords = {
            "packaging": ["包装", "产品", "食品", "饮料", "商品", "品牌", "保质期", "生产日期"],
            "finance_receipt": ["发票", "收据", "票据", "金额", "日期", "收款", "付款"],
            "finance_statement": ["银行", "流水", "账户", "交易明细", "余额", "对账单"],
            "pharmaceutical": ["药品", "医药", "说明书", "批准文号", "用量", "禁忌"],
            "contract": ["合同", "协议", "甲方", "乙方", "条款", "签字", "盖章"],
            "id_card": ["身份证", "证件", "护照", "驾驶证", "姓名", "证号"],
            "logistics": ["快递", "物流", "运单", "寄件", "收件", "包裹"],
            "general_document": ["文档", "文件", "文字", "内容"]
        }
        
        detected_scenario = "general_document"
        max_score = 0
        
        for scenario, keywords in scenario_keywords.items():
            score = sum(1 for kw in keywords if kw.lower() in result_text.lower())
            if score > max_score:
                max_score = score
                detected_scenario = scenario
        
        # 计算置信度（基于关键词匹配数量）
        confidence = min(max_score / 3.0, 1.0)  # 匹配3个关键词即达到100%置信度
        
        logger.info(f"VL场景检测完成: detected_scenario={detected_scenario}, confidence={confidence:.2f}")
        
        return VlScenarioDetectOutput(
            detected_scenario=detected_scenario,
            confidence=confidence,
            detection_reason=result_text[:200]  # 截取前200字符作为检测原因
        )
        
    except Exception as e:
        logger.error(f"VL场景检测失败: {e}")
        # 降级处理：返回通用文档场景
        return VlScenarioDetectOutput(
            detected_scenario="general_document",
            confidence=0.5,
            detection_reason=f"VL检测失败，降级为通用文档场景: {str(e)[:100]}"
        )