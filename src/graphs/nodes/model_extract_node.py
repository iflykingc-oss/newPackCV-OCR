# -*- coding: utf-8 -*-
"""
模型结构化提取节点
使用大语言模型从OCR文本中提取结构化信息
"""

import os
import json
import time
from jinja2 import Template
from typing import Dict, Any, List
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context
from graphs.state import ModelExtractInput, ModelExtractOutput
from langchain_core.messages import SystemMessage, HumanMessage


def model_extract_node(
    state: ModelExtractInput,
    config: RunnableConfig,
    runtime: Runtime[Context]
) -> ModelExtractOutput:
    """
    title: 结构化信息提取
    desc: 使用大语言模型从OCR识别文本中提取结构化信息（如品牌、规格、生产日期等）
    integrations: 大语言模型
    """
    ctx = runtime.context
    
    try:
        # 加载模型配置
        cfg_file = os.path.join(os.getenv("COZE_WORKSPACE_PATH"), config['metadata']['llm_cfg'])
        with open(cfg_file, 'r', encoding='utf-8') as f:
            _cfg = json.load(f)
        
        llm_config = _cfg.get("config", {})
        sp = _cfg.get("sp", "")
        up_tpl = Template(_cfg.get("up", ""))
        
        # 处理多种输入字段名（兼容性）
        ocr_text = state.ocr_text or state.raw_text or state.ocr_raw_result or ""
        
        # 使用自定义提示词或默认提示词
        user_prompt = state.custom_prompt if state.custom_prompt else up_tpl.render({
            "ocr_text": ocr_text,
            "fields": json.dumps(state.template_fields, ensure_ascii=False) if state.template_fields else ""
        })
        
        # 构造消息
        messages = [
            SystemMessage(content=sp),
            HumanMessage(content=user_prompt)
        ]
        
        # 初始化并调用模型
        from coze_coding_dev_sdk.llm import LLMClient
        
        client = LLMClient()
        
        # 设置模型参数
        model_params = {
            "model": llm_config.get("model", state.model_name),
            "temperature": llm_config.get("temperature", 0.1),
            "max_tokens": llm_config.get("max_completion_tokens", 2000),
        }
        
        response = client.invoke(
            messages=messages,
            **model_params
        )
        
        # 解析响应
        result_text = response.content if hasattr(response, 'content') else str(response)
        
        # 尝试解析JSON
        structured_data = {}
        confidence = 0.0
        missing_fields = []
        
        try:
            # 提取JSON部分
            import re
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                structured_data = json.loads(json_match.group())
            else:
                structured_data = json.loads(result_text)
            
            # 计算置信度（基于字段完整性）
            if state.template_fields:
                filled_fields = [f for f in state.template_fields if f in structured_data and structured_data[f]]
                missing_fields = [f for f in state.template_fields if f not in filled_fields]
                confidence = len(filled_fields) / len(state.template_fields) if state.template_fields else 1.0
            else:
                confidence = 0.8  # 默认置信度
                
        except json.JSONDecodeError as e:
            print(f"JSON解析失败，尝试文本提取: {str(e)}")
            # 如果JSON解析失败，尝试从文本中提取键值对
            structured_data = {"raw_extract": result_text}
            confidence = 0.5
        
        print(f"结构化提取完成，置信度: {confidence:.2f}")
        
        return ModelExtractOutput(
            structured_data=structured_data,
            confidence=confidence,
            missing_fields=missing_fields
        )
        
    except Exception as e:
        print(f"模型结构化提取失败: {str(e)}")
        return ModelExtractOutput(
            structured_data={"error": str(e)},
            confidence=0.0,
            missing_fields=[]
        )
