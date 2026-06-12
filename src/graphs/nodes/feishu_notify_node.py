# -*- coding: utf-8 -*-
"""
飞书通知节点
将OCR提取的结构化数据通过飞书机器人发送到群聊
支持卡片消息（字段展示+导出链接）
"""

import os
import json
import logging
from typing import Dict, Any, Optional

import requests
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context
from graphs.state import FeishuNotifyInput, FeishuNotifyOutput

logger = logging.getLogger(__name__)

# 字段中文名映射
FIELD_LABELS: Dict[str, str] = {
    "brand": "🏷️ 品牌",
    "product_name": "📦 产品名称",
    "specification": "⚖️ 规格/净含量",
    "production_date": "📅 生产日期",
    "shelf_life": "⏳ 保质期",
    "expiry_date": "⚠️ 到期日期",
    "manufacturer": "🏭 生产商",
    "ingredients": "🧪 配料",
    "standard": "📋 执行标准",
    "batch_number": "🔢 批号",
    "license_number": "📄 许可证号",
    "storage_condition": "🌡️ 贮存条件",
}

FIELD_ORDER: list[str] = [
    "brand", "product_name", "specification",
    "production_date", "shelf_life", "expiry_date",
    "manufacturer", "ingredients",
    "standard", "batch_number", "license_number", "storage_condition",
]


def _get_webhook_url() -> str:
    """通过 workload identity 获取飞书机器人 webhook URL"""
    from coze_workload_identity import Client
    client = Client()
    credential = client.get_integration_credential("integration-feishu-message")
    cred_data = json.loads(credential)
    webhook_url: str = cred_data["webhook_url"]
    return webhook_url


def _build_structured_section(data: Dict[str, Any]) -> list:
    """构建结构化数据展示内容块（字段值对）"""
    elements: list = []
    for key in FIELD_ORDER:
        value = data.get(key, "N/A")
        if not value or value == "N/A":
            continue
        label = FIELD_LABELS.get(key, key)
        text = f"**{label}**：{value}"
        elements.append({
            "tag": "div",
            "text": {
                "tag": "lark_md",
                "content": text
            }
        })
    return elements


def _build_card_payload(
    structured_data: Dict[str, Any],
    raw_text: str,
    corrected_text: str,
    qa_answer: str,
    export_url: Optional[str] = None,
) -> dict:
    """构建飞书交互式卡片消息"""
    elements: list = []

    # --- 头部摘要 ---
    field_count = sum(1 for k in FIELD_ORDER if structured_data.get(k) and structured_data.get(k) != "N/A")
    total_fields = len(FIELD_ORDER)
    summary = f"📊 共提取 **{field_count}/{total_fields}** 个字段"
    elements.append({
        "tag": "div",
        "text": {"tag": "lark_md", "content": summary}
    })
    elements.append({"tag": "hr"})

    # --- 结构化的字段展示 ---
    section = _build_structured_section(structured_data)
    if section:
        elements.extend(section)
        elements.append({"tag": "hr"})

    # --- 导出链接 ---
    if export_url:
        elements.append({
            "tag": "div",
            "text": {
                "tag": "lark_md",
                "content": f"📎 **完整数据导出**：[点击查看]({export_url})"
            }
        })
        elements.append({"tag": "hr"})

    # --- 语义问答结果（如有） ---
    answer_text = qa_answer or ""
    if answer_text:
        truncated = answer_text[:500] + ("..." if len(answer_text) > 500 else "")
        elements.append({
            "tag": "div",
            "text": {
                "tag": "lark_md",
                "content": f"💬 **智能问答**：\n{truncated}"
            }
        })

    # --- 原始/纠错文本（折叠在备注中） ---
    display_text = corrected_text or raw_text or ""
    if display_text:
        text_preview = display_text[:200].replace("\n", "  \n")
        if len(display_text) > 200:
            text_preview += "..."
        elements.append({
            "tag": "note",
            "elements": [{
                "tag": "plain_text",
                "content": f"📝 OCR原文：{text_preview}"
            }]
        })

    # 标签（纯装饰）
    header_title = "📦 PackCV-OCR 识别结果"
    if field_count == total_fields:
        header_title = "✅ PackCV-OCR 识别完成（全部字段）"

    payload = {
        "msg_type": "interactive",
        "card": {
            "header": {
                "title": {
                    "tag": "plain_text",
                    "content": header_title
                },
                "template": "green" if field_count >= total_fields * 0.8 else "blue"
            },
            "elements": elements
        }
    }
    return payload


def feishu_notify_node(
    state: FeishuNotifyInput,
    config: RunnableConfig,
    runtime: Runtime[Context],
) -> FeishuNotifyOutput:
    """
    title: 飞书通知
    desc: 将OCR提取的结构化数据通过飞书机器人卡片消息推送到群聊。非feishu平台自动跳过。
    integrations: 飞书消息集成
    """
    ctx = runtime.context

    # 仅 platform=feishu 时发送
    if state.platform != "feishu":
        logger.info("platform=%s，跳过飞书推送", state.platform)
        return FeishuNotifyOutput(platform_push_result={"skipped": True, "reason": f"platform={state.platform}"})

    structured_data = state.structured_data or {}
    raw_text = state.raw_text or ""
    corrected_text = state.corrected_text or ""
    qa_answer = state.qa_answer or state.answer or ""
    export_url = state.export_file_url

    try:
        webhook_url = _get_webhook_url()
        payload = _build_card_payload(structured_data, raw_text, corrected_text, qa_answer, export_url)
        logger.info("sending feishu card message (fields=%d/%d)", 
                     sum(1 for k in FIELD_ORDER if structured_data.get(k) and structured_data.get(k) != "N/A"),
                     len(FIELD_ORDER))

        resp = requests.post(webhook_url, json=payload, timeout=30)
        resp.raise_for_status()
        result = resp.json()
        logger.info("feishu push success: %s", result.get("code", result))

        return FeishuNotifyOutput(platform_push_result={
            "success": True,
            "response": result,
        })

    except Exception as e:
        logger.error("feishu push failed: %s", str(e))
        return FeishuNotifyOutput(platform_push_result={
            "success": False,
            "error": str(e),
        })