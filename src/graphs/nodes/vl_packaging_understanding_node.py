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

from graphs.state import VLPackagingInput, VLPackagingOutput

logger = logging.getLogger(__name__)


def vl_packaging_understanding_node(
    state: VLPackagingInput,
    config: RunnableConfig,
    runtime: Runtime[Context]
) -> VLPackagingOutput:
    """
    title: VL多模态包装理解
    desc: 直接使用多模态大模型(豆包Seed-2.0-Pro)理解包装图片，跳过传统OCR管线，端到端提取所有可见信息
    integrations: 大语言模型
    """
    ctx = runtime.context

    # 读取模型配置
    cfg_file = os.path.join(os.getenv("COZE_WORKSPACE_PATH", ""), config['metadata']['llm_cfg'])
    with open(cfg_file, 'r', encoding='utf-8') as f:
        _cfg = json.load(f)

    llm_config = _cfg.get("config", {})
    sp = _cfg.get("sp", "")
    up = _cfg.get("up", "")

    # 获取图片URL
    image_url = state.package_image.url

    # 使用LLMClient调用多模态模型
    client = LLMClient(ctx=ctx)

    # 构造多模态消息
    messages = [
        SystemMessage(content=sp),
        HumanMessage(content=[
            {"type": "text", "text": up},
            {"type": "image_url", "image_url": {"url": image_url}}
        ])
    ]

    try:
        response = client.invoke(
            messages=messages,
            model=llm_config.get("model", "doubao-seed-2-0-pro-260215"),
            temperature=llm_config.get("temperature", 0.05),
            top_p=llm_config.get("top_p", 0.7),
            max_tokens=llm_config.get("max_completion_tokens", 4000),
        )

        # 解析返回文本（兼容str和list类型）
        result_text = ""
        if isinstance(response.content, str):
            result_text = response.content.strip()
        elif isinstance(response.content, list):
            text_parts = []
            for item in response.content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif isinstance(item, str):
                    text_parts.append(item)
            result_text = " ".join(text_parts).strip()
        else:
            result_text = str(response.content)

        # 提取JSON
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            try:
                parsed = json.loads(json_str)
            except json.JSONDecodeError:
                logger.warning("VL理解结果JSON解析失败，使用原始文本")
                parsed = {"raw_extraction": result_text}
        else:
            parsed = {"raw_extraction": result_text}

        return VLPackagingOutput(
            vl_extracted_data=parsed,
            vl_raw_response=result_text,
            vl_success=True
        )

    except Exception as e:
        logger.error(f"VL模型调用失败: {str(e)}")
        return VLPackagingOutput(
            vl_extracted_data={},
            vl_raw_response=f"VL模型调用失败: {str(e)}",
            vl_success=False,
            vl_error=str(e)
        )