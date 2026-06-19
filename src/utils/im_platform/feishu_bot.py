# -*- coding: utf-8 -*-
"""飞书机器人 Adapter - 复用现有webhook，支持事件订阅、签名验证、命令路由"""

import json
import time
import hashlib
import hmac
import base64
import logging
from typing import Any, Dict, List, Optional
import urllib.request
import urllib.error

from .base import IMPlatform, IMMessageType

logger = logging.getLogger(__name__)


class FeishuPlatform(IMPlatform):
    """飞书平台适配"""

    platform_name = "feishu"

    def __init__(self, encrypt_key: str = "", verification_token: str = ""):
        self.encrypt_key = encrypt_key
        self.verification_token = verification_token

    def verify_signature(self, headers: Dict[str, str], body: str) -> bool:
        """飞书事件签名验证：timestamp + nonce → SHA1"""
        timestamp = headers.get("X-Lark-Request-Timestamp", headers.get("X-Lark-Signature-Timestamp", ""))
        nonce = headers.get("X-Lark-Request-Nonce", headers.get("X-Lark-Signature-Nonce", ""))
        signature = headers.get("X-Lark-Signature", "")
        if not (timestamp and nonce and signature):
            return True  # 无加密配置时跳过
        content = f"{timestamp}{nonce}{self.encrypt_key}{body}"
        digest = hashlib.sha1(content.encode("utf-8")).hexdigest()
        return digest == signature

    def parse_event(self, body: str) -> Dict[str, Any]:
        """解析飞书事件订阅回调"""
        try:
            data = json.loads(body)
        except Exception:
            return {"event_type": "unknown", "raw": {}}

        # 1. URL验证
        if data.get("type") == "url_verification":
            return {
                "event_type": "url_verification",
                "challenge": data.get("challenge", ""),
                "raw": data,
            }
        # 2. 事件回调
        header = data.get("header", {})
        event_type = header.get("event_type", "")
        event = data.get("event", {})

        if event_type == "im.message.receive_v1":
            message = event.get("message", {})
            sender = event.get("sender", {})
            sender_id = sender.get("sender_id", {})
            chat_id = message.get("chat_id", "")
            chat_type = message.get("chat_type", "dm")
            msg_type = message.get("message_type", "text")
            content_raw = message.get("content", "{}")
            try:
                content = json.loads(content_raw) if isinstance(content_raw, str) else content_raw
            except Exception:
                content = {"text": content_raw}
            return {
                "event_id": header.get("event_id", ""),
                "event_type": "message",
                "user_id": sender_id.get("user_id", "") if isinstance(sender_id, dict) else str(sender_id),
                "chat_id": chat_id,
                "chat_type": chat_type,
                "message_type": msg_type,
                "content": content,
                "raw": data,
            }
        return {"event_type": "unknown", "event_type_detail": event_type, "raw": data}

    def build_text_message(self, text: str, at_user_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        content_text = text
        if at_user_ids:
            at_str = "".join([f'<at user_id="{uid}">@用户</at> ' for uid in at_user_ids])
            content_text = f"{at_str}{text}"
        return {"msg_type": "text", "content": {"text": content_text}}

    def build_card_message(
        self,
        title: str,
        content: str,
        actions: Optional[List[Dict[str, Any]]] = None,
        template: str = "blue"
    ) -> Dict[str, Any]:
        elements: List[Dict[str, Any]] = [{
            "tag": "div",
            "text": {"tag": "lark_md", "content": content}
        }]
        if actions:
            elements.append({"tag": "action", "actions": actions})
        return {
            "msg_type": "interactive",
            "card": {
                "header": {
                    "title": {"tag": "plain_text", "content": title},
                    "template": template
                },
                "elements": elements
            }
        }

    def build_markdown_message(self, title: str, text: str) -> Dict[str, Any]:
        return {
            "msg_type": "interactive",
            "card": {
                "header": {
                    "title": {"tag": "plain_text", "content": title},
                    "template": "blue"
                },
                "elements": [{
                    "tag": "div",
                    "text": {"tag": "lark_md", "content": text}
                }]
            }
        }

    def build_ack_message(self, text: str) -> Dict[str, Any]:
        return {
            "msg_type": "interactive",
            "card": {
                "header": {
                    "title": {"tag": "plain_text", "content": "⏳ 正在处理"},
                    "template": "grey"
                },
                "elements": [{
                    "tag": "div",
                    "text": {"tag": "lark_md", "content": text}
                }]
            }
        }

    def send_webhook(self, webhook_url: str, payload: Dict[str, Any], timeout: int = 10) -> Dict[str, Any]:
        import requests  # type: ignore
        try:
            resp = requests.post(webhook_url, json=payload, timeout=timeout)
            return {"ok": resp.status_code == 200, "status": resp.status_code, "body": resp.text}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def extract_text_from_event(self, event: Dict[str, Any]) -> str:
        if event.get("event_type") != "message":
            return ""
        content = event.get("content", {})
        if isinstance(content, dict):
            return (content.get("text") or "").strip()
        return str(content).strip()
