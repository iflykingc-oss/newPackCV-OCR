# -*- coding: utf-8 -*-
"""
语义问答节点
使用大语言模型对OCR结果进行问答分析
"""

import os
import re
import json
import time
from jinja2 import Template
from typing import Dict, Any
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context
from graphs.state import QaAnswerInput, QaAnswerOutput
from langchain_core.messages import SystemMessage, HumanMessage
try:
    from coze_coding_dev_sdk import LLMClient
except ImportError:
    LLMClient = None


def qa_answer_node(
    state: QaAnswerInput,
    config: RunnableConfig,
    runtime: Runtime[Context]
) -> QaAnswerOutput:
    """
    title: 语义问答
    desc: 使用大语言模型对OCR识别结果进行语义问答分析
    integrations: 大语言模型
    """
    ctx = runtime.context

    # 安全获取OCR文本
    ocr_text = state.ocr_text or state.raw_text or state.ocr_raw_result or ""
    question = state.user_question or "请分析这个产品的完整信息"

    try:
        # 加载模型配置
        cfg_file = os.path.join(os.getenv("COZE_WORKSPACE_PATH"), config['metadata']['llm_cfg'])
        with open(cfg_file, 'r', encoding='utf-8') as f:
            _cfg = json.load(f)

        llm_config = _cfg.get("config", {})
        sp = _cfg.get("sp", "")
        up_tpl = Template(_cfg.get("up", ""))

        # 构造提示词
        user_prompt = up_tpl.render({
            "ocr_text": ocr_text,
            "question": question,
            "structured_data": json.dumps(state.structured_data, ensure_ascii=False) if state.structured_data else "无"
        })

        # 构造消息
        messages = [
            SystemMessage(content=sp),
            HumanMessage(content=user_prompt)
        ]

        # 初始化并调用模型
        client = LLMClient(ctx=ctx)

        response = client.invoke(
            messages=messages,
            model=llm_config.get("model", state.model_name),
            temperature=llm_config.get("temperature", 0.3),
            max_tokens=llm_config.get("max_completion_tokens", 2000)
        )

        # 解析响应 - 安全获取文本内容
        answer = ""
        if response and hasattr(response, 'content'):
            content = response.content
            if isinstance(content, str):
                answer = content
            elif isinstance(content, list):
                if content and isinstance(content[0], str):
                    answer = " ".join(content)
                else:
                    text_parts = [item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"]
                    answer = " ".join(text_parts).strip()
            else:
                answer = str(content)

        print(f"语义问答完成")

        return QaAnswerOutput(
            answer=answer,
            confidence=0.8
        )

    except Exception as e:
        error_msg = f"语义问答失败: {str(e)}"
        print(error_msg)
        # 降级到简单拼接
        simple_answer = f"基于OCR文本的分析结果：\n{ocr_text}"
        if state.structured_data:
            simple_answer += f"\n\n结构化数据：\n{json.dumps(state.structured_data, ensure_ascii=False, indent=2)}"

        return QaAnswerOutput(
            answer=simple_answer,
            confidence=0.3
        )
