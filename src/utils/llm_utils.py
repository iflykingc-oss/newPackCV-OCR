#!/usr/bin/env python3
"""
LLM调用工具 - 集成LangSmith追踪 + 限流器 + 指标采集
所有LLM节点复用此模块
"""
import os
import time
import json
import logging
from typing import Optional, Dict, Any
from jinja2 import Template
from langchain_core.messages import SystemMessage, HumanMessage
from coze_coding_dev_sdk import LLMClient

# LangSmith追踪
try:
    from langsmith import traceable
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    def traceable(name=None):
        def decorator(func):
            return func
        return decorator

# 限流器
try:
    from utils.rate_limiter import get_rate_limiter
    RATE_LIMITER_AVAILABLE = True
except ImportError:
    RATE_LIMITER_AVAILABLE = False

# 指标采集
try:
    from utils.metrics_utils import record_llm_call
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

logger = logging.getLogger(__name__)


@traceable(name="llm_multimodal_call", run_type="llm")
def call_llm_multimodal(
    prompt: str,
    image_url: str,
    model: str,
    sp: str = "",
    up: str = "",
    temperature: float = 0.05,
    max_tokens: int = 4000,
    fallback_chain: Optional[list] = None,
    node_name: str = "unknown",
    ctx=None,
) -> Dict[str, Any]:
    """
    生产级多模态LLM调用
    - LangSmith追踪
    - 限流器检查
    - Fallback链
    - 指标采集
    - 错误处理
    """
    start_time = time.time()
    api_call_count = 0
    total_cost = 0.0

    # 1. 限流检查
    if RATE_LIMITER_AVAILABLE:
        limiter = get_rate_limiter()
        if not limiter.allow_request(model):
            raise Exception(f"模型 {model} 触发限流，请稍后重试")

    # 2. 渲染Prompt
    if up:
        up_tpl = Template(up)
        rendered_up = up_tpl.render({
            "image_url": image_url,
            "user_prompt": prompt
        })
    else:
        rendered_up = prompt

    # 3. 构造消息
    messages = []
    if sp:
        messages.append(SystemMessage(content=sp))
    messages.append(HumanMessage(content=[
        {"type": "text", "text": rendered_up},
        {"type": "image_url", "image_url": {"url": image_url}}
    ]))

    # 4. Fallback链调用
    models_to_try = [model] + (fallback_chain or [])
    last_error = None

    for attempt_model in models_to_try:
        try:
            client = LLMClient(ctx=ctx) if ctx else LLMClient()
            response = client.invoke(
                model=attempt_model,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=max_tokens
            )

            api_call_count += 1
            duration = time.time() - start_time

            # 5. 记录指标
            if METRICS_AVAILABLE:
                record_llm_call(
                    node=node_name,
                    model=attempt_model,
                    duration=duration,
                    success=True,
                    tokens=getattr(response, 'usage', {}).get('total_tokens', 0) if hasattr(response, 'usage') else 0
                )

            content = response.content if hasattr(response, 'content') else str(response)

            return {
                "content": content,
                "model_used": attempt_model,
                "duration_s": duration,
                "api_call_count": api_call_count,
                "fallback_used": attempt_model != model,
                "tokens_used": getattr(response, 'usage', {}).get('total_tokens', 0) if hasattr(response, 'usage') else 0
            }

        except Exception as e:
            logger.warning(f"模型 {attempt_model} 调用失败: {e}")
            last_error = e
            continue

    # 所有模型都失败
    duration = time.time() - start_time
    if METRICS_AVAILABLE:
        record_llm_call(
            node=node_name,
            model=model,
            duration=duration,
            success=False,
            error=str(last_error)
        )
    raise Exception(f"所有Fallback模型均失败，最后错误: {last_error}")


def parse_llm_json(content: str) -> Dict[str, Any]:
    """解析LLM返回的JSON内容（处理Markdown包裹）"""
    text = content.strip()

    # 去除Markdown代码块
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # 尝试提取JSON片段
        import re
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except:
                pass
        return {"error": "json_parse_failed", "raw": content[:500]}
