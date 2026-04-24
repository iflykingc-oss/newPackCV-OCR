# -*- coding: utf-8 -*-
"""
智能纠错节点
使用大语言模型对OCR识别结果进行智能纠错
"""

import os
import json
import time
from jinja2 import Template
from typing import Dict, Any, List
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context
from graphs.state import CorrectTextInput, CorrectTextOutput
from langchain_core.messages import SystemMessage, HumanMessage


def correct_text_node(
    state: CorrectTextInput,
    config: RunnableConfig,
    runtime: Runtime[Context]
) -> CorrectTextOutput:
    """
    title: 智能纠错
    desc: 使用大语言模型对OCR识别结果进行智能纠错，修复错别字、漏字等问题
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
        
        # 构造提示词
        correction_rules_str = json.dumps(state.correction_rules, ensure_ascii=False) if state.correction_rules else "无特殊规则"
        
        user_prompt = up_tpl.render({
            "ocr_text": ocr_text,
            "correction_rules": correction_rules_str
        })
        
        # 构造消息
        messages = [
            SystemMessage(content=sp),
            HumanMessage(content=user_prompt)
        ]
        
        # 初始化并调用模型
        from coze_coding_dev_sdk.llm import LLMClient
        
        client = LLMClient()
        
        response = client.invoke(
            messages=messages,
            model=llm_config.get("model", state.model_name),
            temperature=llm_config.get("temperature", 0.1),
            max_tokens=llm_config.get("max_completion_tokens", 2000)
        )
        
        # 解析响应
        result_text = response.content if hasattr(response, 'content') else str(response)
        
        # 尝试解析纠错结果
        corrected_text = state.ocr_text  # 默认返回原文
        changes = []
        
        try:
            import re
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                
                if "corrected_text" in result:
                    corrected_text = result["corrected_text"]
                
                if "changes" in result:
                    changes = result["changes"]
                    
                # 如果没有changes但有corrected_text，自动对比
                if not changes and "corrected_text" in result:
                    original_lines = state.ocr_text.split('\n')
                    corrected_lines = corrected_text.split('\n')
                    for i, (orig, corr) in enumerate(zip(original_lines, corrected_lines)):
                        if orig != corr:
                            changes.append({
                                "index": i,
                                "original": orig,
                                "corrected": corr
                            })
            else:
                # 如果无法解析JSON，直接使用返回文本作为纠错结果
                corrected_text = result_text
                # 简单对比
                if state.ocr_text != corrected_text:
                    changes.append({
                        "original": state.ocr_text,
                        "corrected": corrected_text
                    })
                    
        except Exception as e:
            print(f"解析纠错结果失败，直接使用模型输出: {str(e)}")
            corrected_text = result_text
        
        correction_count = len(changes)
        
        print(f"智能纠错完成，修正{correction_count}处")
        
        return CorrectTextOutput(
            corrected_text=corrected_text,
            changes=changes,
            correction_count=correction_count
        )
        
    except Exception as e:
        print(f"智能纠错失败: {str(e)}")
        return CorrectTextOutput(
            corrected_text=state.ocr_text,
            changes=[],
            correction_count=0
        )
