# -*- coding: utf-8 -*-
"""企业微信机器人 Adapter - 智能机器人/群机器人"""

import json
import time
import hashlib
import logging
from typing import Any, Dict, List, Optional

from .base import IMPlatform, IMMessageType

logger = logging.getLogger(__name__)


class WeComPlatform(IMPlatform):
    """企业微信平台适配"""

    platform_name = "wecom"

    def __init__(self, token: str = "", encoding_aes_key: str = ""):
        self.token = token
        self.encoding_aes_key = encoding_aes_key

    def verify_signature(self, headers: Dict[str, str], body: str) -> bool:
        """企业微信签名验证：msg_signature + timestamp + nonce + echostr"""
        msg_signature = headers.get("msg_signature", "")
        timestamp = headers.get("timestamp", "")
        nonce = headers.get("nonce", "")
        if not (msg_signature and timestamp and nonce and self.token):
            return True
        # SHA1加密: [token, timestamp, nonce, encrypt] 排序后拼接
        params = sorted([self.token, timestamp, nonce])
        sha1 = hashlib.sha1("".join(params).encode("utf-8")).hexdigest()
        return sha1 == msg_signature

    def parse_event(self, body: str) -> Dict[str, Any]:
        """企业微信事件回调"""
        try:
            data = json.loads(body)
        except Exception:
            return {"event_type": "unknown", "raw": {}}
        # 加密消息体（需要解密）
        if "Encrypt" in data:
            return {
                "event_type": "encrypted",
                "encrypt": data.get("Encrypt", ""),
                "raw": data,
            }
        # URL验证
        if data.get("msgtype") == "event" and data.get("event") == "weapp_event":
            echostr = data.get("echostr", "")
            if echostr:
                return {"event_type": "url_verification", "challenge": echostr, "raw": data}
        # 文本消息
        if data.get("msgtype") == "text":
            return {
                "event_type": "message",
                "message_type": "text",
                "user_id": data.get("from", ""),
                "chat_id": data.get("chat_id", ""),
                "chat_type": data.get("chat_type", "single"),
                "content": {"text": data.get("text", {}).get("content", "")},
                "raw": data,
            }
        # 图片消息
        if data.get("msgtype") == "image":
            return {
                "event_type": "message",
                "message_type": "image",
                "user_id": data.get("from", ""),
                "chat_id": data.get("chat_id", ""),
                "chat_type": data.get("chat_type", "single"),
                "content": {"image_url": data.get("image", {}).get("url", "")},
                "raw": data,
            }
        return {"event_type": "unknown", "raw": data}

    def build_text_message(self, text: str, at_user_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        msg: Dict[str, Any] = {
            "msgtype": "text",
            "text": {"content": text}
        }
        if at_user_ids:
            msg["text"]["mentioned_list"] = at_user_ids
        return msg

    def build_card_message(
        self,
        title: str,
        content: str,
        actions: Optional[List[Dict[str, Any]]] = None,
        template: str = "blue"
    ) -> Dict[str, Any]:
        # 企业微信 textcard
        return {
            "msgtype": "textcard",
            "textcard": {
                "title": title,
                "description": content,
                "url": (actions[0].get("url", "https://example.com") if actions else "https://example.com"),
                "btntxt": "查看详情"
            }
        }

    def build_markdown_message(self, title: str, text: str) -> Dict[str, Any]:
        return {
            "msgtype": "markdown",
            "markdown": {
                "content": text
            }
        }

    def build_ack_message(self, text: str) -> Dict[str, Any]:
        return {
            "msgtype": "markdown",
            "markdown": {
                "content": f"## ⏳ 正在处理\n\n{text}"
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
