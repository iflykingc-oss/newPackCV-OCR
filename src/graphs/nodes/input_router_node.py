# -*- coding: utf-8 -*-
"""
输入类型路由节点
判断输入是图片还是文档，分流到不同处理管线
V5.9 新增：支持PDF/DOCX/PPTX/XLSX文档输入
"""

import os
import re
import logging
from typing import Optional
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context
from graphs.state import InputTypeRouteInput, InputTypeRouteOutput

logger = logging.getLogger(__name__)

# 文档文件后缀
DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".pptx", ".xlsx", ".doc", ".ppt", ".xls"}
# 图片文件后缀
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".gif"}


def _detect_type_by_url(url: str) -> tuple:
    """根据URL检测文件类型

    Args:
        url: 文件URL

    Returns:
        (input_type, file_type, confidence, reason)
    """
    url_lower = url.lower().strip()

    # 提取文件后缀
    match = re.search(r'\.([a-z0-9]+)(?:\?|#|$)', url_lower)
    if not match:
        return "image", "", 0.5, "无法判断后缀，默认走图片管线"

    ext = f".{match.group(1)}"

    if ext in DOCUMENT_EXTENSIONS:
        return "document", ext, 0.95, f"检测到文档格式: {ext}"

    if ext in IMAGE_EXTENSIONS:
        return "image", ext, 0.95, f"检测到图片格式: {ext}"

    # 未知格式，默认走图片管线
    return "image", ext, 0.3, f"未识别格式({ext})，默认走图片管线"


def input_router_node(
    state: InputTypeRouteInput,
    config: RunnableConfig,
    runtime: Runtime[Context],
) -> InputTypeRouteOutput:
    """
    title: 输入类型路由
    desc: 判断输入是图片还是文档（PDF/DOCX/PPTX/XLSX），分流到不同处理管线
    integrations: 
    """
    ctx = runtime.context

    # 确定文件URL
    file_url = state.file_url or ""
    if state.package_image and state.package_image.url:
        file_url = state.package_image.url

    if not file_url:
        return InputTypeRouteOutput(
            input_type="unsupported",
            file_url="",
            file_type="",
            confidence=0.0,
            reason="未提供有效的文件输入",
        )

    input_type, file_type, confidence, reason = _detect_type_by_url(file_url)

    logger.info(
        "输入类型路由: type=%s, ext=%s, confidence=%.2f, url=%s",
        input_type, file_type, confidence, file_url,
    )

    return InputTypeRouteOutput(
        input_type=input_type,
        file_url=file_url,
        file_type=file_type,
        confidence=confidence,
        reason=reason,
    )