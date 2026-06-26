import os
import json
import re
import logging
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context
try:
    from coze_coding_dev_sdk import LLMClient
except ImportError:
    LLMClient = None
from langchain_core.messages import SystemMessage, HumanMessage
from jinja2 import Template

from graphs.state import KnowledgeInferenceInput, KnowledgeInferenceOutput

logger = logging.getLogger(__name__)


def knowledge_inference_node(
    state: KnowledgeInferenceInput,
    config: RunnableConfig,
    runtime: Runtime[Context]
) -> KnowledgeInferenceOutput:
    """
    title: 知识图谱推理补全
    desc: 基于已提取的产品信息，通过知识推理补充缺失字段（保质期、存储条件等通用信息），并验证已有字段合理性
    integrations: 大语言模型
    """
    ctx = runtime.context

    # 读取模型配置
    cfg_file = os.path.join(os.getenv("COZE_WORKSPACE_PATH", ""), config['metadata']['llm_cfg'])
    with open(cfg_file, 'r', encoding='utf-8') as f:
        _cfg = json.load(f)

    llm_config = _cfg.get("config", {})
    sp = _cfg.get("sp", "")
    up_template = _cfg.get("up", "")

    # 准备输入数据（字段名与KnowledgeInferenceInput对齐）
    extracted_data = state.structured_data or {}
    raw_text = state.raw_text or ""

    # 渲染模板
    up = Template(up_template).render(
        extracted_data=json.dumps(extracted_data, ensure_ascii=False, indent=2),
        raw_text=raw_text,
        product_type=state.product_type
    )

    # 使用LLMClient调用大模型
    client = LLMClient(ctx=ctx)
    messages = [
        SystemMessage(content=sp),
        HumanMessage(content=up)
    ]

    try:
        response = client.invoke(
            messages=messages,
            model=llm_config.get("model", "kimi-k2-5-260127"),
            temperature=llm_config.get("temperature", 0.1),
            top_p=llm_config.get("top_p", 0.95),
            max_tokens=llm_config.get("max_completion_tokens", 2000),
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
                logger.warning("知识推理结果JSON解析失败，使用原始文本")
                parsed = {"raw_inference": result_text}
        else:
            parsed = {"raw_inference": result_text}

        inferred_fields = parsed.get("inferred_fields", [])
        validations = parsed.get("validations", [])
        product_type_val = parsed.get("product_type", state.product_type)

        return KnowledgeInferenceOutput(
            inferred_fields=inferred_fields,
            validation_results=validations,
            inferred_product_type=product_type_val,
            inference_raw_response=result_text
        )

    except Exception as e:
        logger.error(f"知识推理调用失败: {str(e)}")
        return KnowledgeInferenceOutput(
            inferred_fields=[],
            validation_results=[{"field": "整体", "status": "error", "note": str(e)}],
            inferred_product_type=state.product_type,
            inference_raw_response=f"知识推理失败: {str(e)}"
        )