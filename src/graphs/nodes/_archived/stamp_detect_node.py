# -*- coding: utf-8 -*-
"""
印章/公章检测节点
使用VL多模态模型检测和识别图片中的印章
V5.9 新增
"""

import os
import json
import logging
from typing import List, Dict, Any
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context
from graphs.state import StampDetectInput, StampDetectOutput
from jinja2 import Template

logger = logging.getLogger(__name__)


def stamp_detect_node(
    state: StampDetectInput,
    config: RunnableConfig,
    runtime: Runtime[Context],
) -> StampDetectOutput:
    """
    title: 印章/公章检测
    desc: 使用多模态大模型检测图片中是否包含印章/公章，识别印章类型和文字内容（合同/金融场景）
    integrations: 大语言模型
    """
    ctx = runtime.context

    # 读取LLM配置
    cfg_file = os.path.join(os.getenv("COZE_WORKSPACE_PATH", ""), config["metadata"]["llm_cfg"])
    with open(cfg_file, "r") as fd:
        _cfg = json.load(fd)

    llm_config = _cfg.get("config", {})
    sp = _cfg.get("sp", "")
    up = _cfg.get("up", "")

    # 渲染用户提示词
    up_tpl = Template(up)
    user_prompt = up_tpl.render({})

    # 调用多模态LLM检测印章
    from coze_coding_dev_sdk import LLMClient
    from langchain_core.messages import SystemMessage, HumanMessage

    client = LLMClient(llm_config)

    # 构建多模态消息（图片+文本）
    image_url = state.package_image.url
    msg = HumanMessage(
        content=[
            {"type": "text", "text": user_prompt},
            {
                "type": "image_url",
                "image_url": {"url": image_url, "detail": "high"},
            },
        ]
    )

    messages = [SystemMessage(content=sp), msg]
    response = client.invoke(messages)

    # 解析响应（期望JSON格式）
    raw_content = response.content
    if isinstance(raw_content, list):
        # 多模态响应可能返回list，提取文本部分
        texts = [item for item in raw_content if isinstance(item, str)]
        content = " ".join(texts) if texts else json.dumps(raw_content)
    else:
        content = str(raw_content)
    # 提取JSON部分
    json_match = None
    import re
    json_match = re.search(r'\{[\s\S]*\}', content)

    if json_match:
        try:
            result = json.loads(json_match.group())
        except json.JSONDecodeError:
            result = {"stamps": [], "total": 0}
    else:
        # 非JSON响应，尝试结构化解析
        has_stamp = "是" in content or "有" in content or "检测到" in content
        result = {"stamps": [], "total": 0} if not has_stamp else {
            "stamps": [{"type": "unknown", "confidence": 0.5, "text": content[:200]}],
            "total": 1,
        }

    stamps = result.get("stamps", [])
    total = result.get("total", len(stamps))

    logger.info("印章检测完成: 共%d个印章", total)

    return StampDetectOutput(
        stamps=stamps,
        total_found=total,
        has_stamp=total > 0,
    )