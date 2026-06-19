# -*- coding: utf-8 -*-
"""钉钉机器人 Adapter - 自定义机器人webhook + 加密签名"""

import json
import time
import hashlib
import hmac
import base64
import logging
from typing import Any, Dict, List, Optional

from .base import IMPlatform, IMMessageType

logger = logging.getLogger(__name__)


class DingTalkPlatform(IMPlatform):
    """钉钉平台适配（自定义机器人模式）"""

    platform_name = "dingtalk"

    def __init__(self, secret: str = ""):
        self.secret = secret

    def verify_signature(self, headers: Dict[str, str], body: str) -> bool:
        """钉钉签名验证：timestamp + secret → SHA256"""
        timestamp = headers.get("timestamp", "")
        sign = headers.get("sign", "")
        if not (timestamp and sign):
            return True  # 旧版无加密可跳过
        if not self.secret:
            return False
        string_to_sign = f"{timestamp}\n{self.secret}"
        hmac_code = hmac.new(
            self.secret.encode("utf-8"),
            string_to_sign.encode("utf-8"),
            digestmod=hashlib.sha256
        ).digest()
        expected = urllib_quote_plus(base64.b64encode(hmac_code))
        return expected == sign

    def parse_event(self, body: str) -> Dict[str, Any]:
        """钉钉事件回调（Stream模式/机器人回调）"""
        try:
            data = json.loads(body)
        except Exception:
            return {"event_type": "unknown", "raw": {}}
        # 1. Stream模式加密消息
        if "encrypt" in data:
            return {
                "event_type": "encrypted",
                "encrypt": data.get("encrypt", ""),
                "raw": data,
            }
        # 2. 文本/图片消息回调
        msg_type = data.get("msgtype", "")
        text_content = data.get("text", {}).get("content", "")
        if msg_type == "text":
            return {
                "event_type": "message",
                "message_type": "text",
                "user_id": data.get("senderId", ""),
                "chat_id": data.get("conversationId", data.get("chatId", "")),
                "chat_type": data.get("conversationType", "1") == "1" and "dm" or "group",
                "content": {"text": text_content},
                "raw": data,
            }
        if msg_type == "picture":
            return {
                "event_type": "message",
                "message_type": "image",
                "user_id": data.get("senderId", ""),
                "chat_id": data.get("conversationId", data.get("chatId", "")),
                "chat_type": data.get("conversationType", "1") == "1" and "dm" or "group",
                "content": {"image_url": data.get("pictureUrl", "")},
                "raw": data,
            }
        return {"event_type": "unknown", "raw": data}

    def build_text_message(self, text: str, at_user_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        msg: Dict[str, Any] = {
            "msgtype": "text",
            "text": {"content": text}
        }
        if at_user_ids:
            msg["at"] = {"atUserIds": at_user_ids}
        return msg

    def build_card_message(
        self,
        title: str,
        content: str,
        actions: Optional[List[Dict[str, Any]]] = None,
        template: str = "blue"
    ) -> Dict[str, Any]:
        # 钉钉ActionCard
        return {
            "msgtype": "actionCard",
            "actionCard": {
                "title": title,
                "text": content,
                "singleTitle": "查看详情" if actions else "",
                "singleURL": (actions[0].get("url", "") if actions else "")
            }
        }

    def build_markdown_message(self, title: str, text: str) -> Dict[str, Any]:
        return {
            "msgtype": "markdown",
            "markdown": {
                "title": title,
                "text": text
            }
        }

    def build_ack_message(self, text: str) -> Dict[str, Any]:
        return {
            "msgtype": "markdown",
            "markdown": {
                "title": "⏳ 正在处理",
                "text": f"## ⏳ 正在处理\n\n{text}"
            }
        }

    def send_webhook(self, webhook_url: str, payload: Dict[str, Any], timeout: int = 10) -> Dict[str, Any]:
        import requests  # type: ignore
        # 钉钉加密模式：加签
        if self.secret:
            timestamp = str(round(time.time() * 1000))
            string_to_sign = f"{timestamp}\n{self.secret}"
            hmac_code = hmac.new(
                self.secret.encode("utf-8"),
                string_to_sign.encode("utf-8"),
                digestmod=hashlib.sha256
            ).digest()
            sign = urllib_quote_plus(base64.b64encode(hmac_code))
            sep = "&" if "?" in webhook_url else "?"
            url = f"{webhook_url}{sep}timestamp={timestamp}&sign={sign}"
        else:
            url = webhook_url
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
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


def urllib_quote_plus(s: bytes) -> str:
    import urllib.parse
    return urllib.parse.quote_plus(s)
