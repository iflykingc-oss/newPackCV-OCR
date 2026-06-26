<<<<<<< HEAD
#!/usr/bin/env python3
"""QA问答节点 - 用户问答"""
import os
import json
import logging
from typing import Dict, Any
from jinja2 import Template
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context
from coze_coding_dev_sdk import LLMClient
from langchain_core.messages import SystemMessage, HumanMessage
from graphs.state import QaAnswerInput, QaAnswerOutput

logger = logging.getLogger(__name__)
=======
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
from coze_coding_dev_sdk import LLMClient
>>>>>>> origin/main


def qa_answer_node(
    state: QaAnswerInput,
    config: RunnableConfig,
    runtime: Runtime[Context]
) -> QaAnswerOutput:
    """
<<<<<<< HEAD
    title: 用户问答
    desc: 根据提取的信息调用LLM回答用户问题
    integrations: 大语言模型
    """
    ctx = runtime.context
    
    # 从配置文件读取LLM配置
    cfg_path = config.get("metadata", {}).get("llm_cfg", "")
    llm_config: Dict[str, Any] = {}
    sp: str = ""
    up: str = ""
    
    if cfg_path:
        full_cfg_path = os.path.join(os.getenv("COZE_WORKSPACE_PATH", ""), cfg_path)
        try:
            with open(full_cfg_path, 'r', encoding='utf-8') as f:
                llm_cfg = json.load(f)
                llm_config = llm_cfg.get("config", {})
                sp = llm_cfg.get("sp", "")
                up = llm_cfg.get("up", "")
        except Exception as e:
            logger.warning(f"无法读取配置文件 {full_cfg_path}: {e}")
    
    structured_data = state.structured_data
    user_question = state.user_question
    
    # 获取模型配置参数
    model: str = llm_config.get("model", "doubao-seed-1-8-251228")
    temperature: float = llm_config.get("temperature", 0.7)
    
    # 使用Jinja2渲染用户提示词
    if up:
        up_tpl = Template(up)
        user_prompt_content = up_tpl.render({
            "structured_data": json.dumps(structured_data, ensure_ascii=False),
            "user_question": user_question
        })
    else:
        user_prompt_content = f"""根据以下结构化信息回答用户问题。

结构化数据：
{json.dumps(structured_data, ensure_ascii=False)}

用户问题：{user_question}

请给出准确、专业的回答。"""

    # 构建消息
    messages: list = []
    if sp:
        messages.append(SystemMessage(content=sp))
    messages.append(HumanMessage(content=user_prompt_content))
    
    # 调用LLM生成真实回答
    try:
        client = LLMClient(ctx=ctx)
        response = client.invoke(
            messages=messages,
            model=model,
            temperature=temperature
        )
        
        # 处理响应内容（可能是str或list）
        content = response.content
        if isinstance(content, str):
            qa_answer = content
        elif isinstance(content, list):
            # 处理多模态响应
            text_parts: list = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif isinstance(item, str):
                    text_parts.append(item)
            qa_answer = " ".join(text_parts)
        else:
            qa_answer = str(content)
            
        logger.info(f"QA问答完成: 模型={model}, 问题={user_question[:50]}...")
        
    except Exception as e:
        logger.error(f"LLM调用失败: {e}")
        # 降级处理：基于结构化数据生成基础回答
        qa_answer = f"根据提取的信息：{json.dumps(structured_data, ensure_ascii=False)}，针对您的问题'{user_question}'，请参考上述数据进行进一步分析。"
    
    return QaAnswerOutput(qa_answer=qa_answer)
=======
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
>>>>>>> origin/main
