# -*- coding: utf-8 -*-
"""
语义问答节点
使用大语言模型基于OCR识别结果回答用户问题
"""

import os
import json
import time
from typing import Dict, Any, List
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context
from graphs.state import QaAnswerInput, QaAnswerOutput
from langchain_core.messages import SystemMessage, HumanMessage


def qa_answer_node(
    state: QaAnswerInput,
    config: RunnableConfig,
    runtime: Runtime[Context]
) -> QaAnswerOutput:
    """
    title: 语义问答
    desc: 基于OCR识别结果和结构化数据，使用大语言模型回答用户问题
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
        up = _cfg.get("up", "")
        
        # 构造上下文
        context_parts = []
        
        # 处理多种输入字段名（兼容性）
        ocr_text = state.ocr_text or state.raw_text or state.ocr_raw_result or ""
        
        # 添加OCR文本
        if ocr_text:
            context_parts.append(f"OCR识别文本：\n{ocr_text}\n")
        
        # 添加结构化数据
        if state.structured_data:
            context_parts.append(f"结构化数据：\n{json.dumps(state.structured_data, ensure_ascii=False, indent=2)}\n")
        
        context = "\n".join(context_parts)
        
        # 构造用户消息
        user_message = f"{up}\n\n上下文信息：\n{context}\n\n用户问题：{state.user_question}"
        
        # 构造消息
        messages = [
            SystemMessage(content=sp),
            HumanMessage(content=user_message)
        ]
        
        # 初始化并调用模型
        from coze_coding_dev_sdk.llm import LLMClient
        
        client = LLMClient()
        
        response = client.invoke(
            messages=messages,
            model=llm_config.get("model", state.model_name),
            temperature=llm_config.get("temperature", 0.3),
            max_tokens=llm_config.get("max_completion_tokens", 1500)
        )
        
        # 解析响应
        answer = response.content if hasattr(response, 'content') else str(response)
        
        # 提取参考来源（如果有）
        references = []
        if "参考" in answer or "来源" in answer:
            import re
            ref_match = re.search(r'参考[:：](.+)', answer)
            if ref_match:
                references.append(ref_match.group(1).strip())
        
        # 默认置信度
        confidence = 0.85
        
        print(f"语义问答完成，答案长度: {len(answer)}")
        
        return QaAnswerOutput(
            answer=answer,
            confidence=confidence,
            references=references
        )
        
    except Exception as e:
        print(f"语义问答失败: {str(e)}")
        return QaAnswerOutput(
            answer=f"抱歉，回答问题时出现错误: {str(e)}",
            confidence=0.0,
            references=[]
        )
