# -*- coding: utf-8 -*-
"""
IM Platform Adapter 抽象层 - V5.3
统一封装飞书/钉钉/企业微信三个平台的消息格式、回调处理、签名验证
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import json
import hashlib
import base64
import hmac
import time
import logging
import urllib.parse

logger = logging.getLogger(__name__)


class IMMessageType:
    """统一消息类型"""
    TEXT = "text"
    MARKDOWN = "markdown"
    CARD = "card"
    NEWS = "news"
    IMAGE = "image"
    FILE = "file"


class IMPlatform(ABC):
    """IM平台抽象基类"""

    platform_name: str = "unknown"

    @abstractmethod
    def verify_signature(self, headers: Dict[str, str], body: str) -> bool:
        """验证回调签名"""
        pass

    @abstractmethod
    def parse_event(self, body: str) -> Dict[str, Any]:
        """解析回调事件为统一格式
        Returns:
            {
                "event_id": str,
                "event_type": "url_verification" | "message" | "unknown",
                "challenge": str (仅验证事件),
                "user_id": str,
                "chat_id": str,
                "chat_type": "group" | "dm",
                "message_type": "text" | "image" | "file",
                "content": str (text) | {"image_key": str} (image),
                "raw": dict (原始事件)
            }
        """
        pass

    @abstractmethod
    def build_text_message(self, text: str, at_user_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """构造文本消息payload"""
        pass

    @abstractmethod
    def build_card_message(
        self,
        title: str,
        content: str,
        actions: Optional[List[Dict[str, Any]]] = None,
        template: str = "blue"
    ) -> Dict[str, Any]:
        """构造卡片消息payload"""
        pass

    @abstractmethod
    def build_markdown_message(self, title: str, text: str) -> Dict[str, Any]:
        """构造Markdown消息payload"""
        pass

    @abstractmethod
    def build_ack_message(self, text: str) -> Dict[str, Any]:
        """构造确认/响应消息payload"""
        pass

    @abstractmethod
    def send_webhook(self, webhook_url: str, payload: Dict[str, Any], timeout: int = 10) -> Dict[str, Any]:
        """通过webhook发送消息"""
        pass

    @abstractmethod
    def extract_text_from_event(self, event: Dict[str, Any]) -> str:
        """从事件中提取纯文本内容（用于命令路由）"""
        pass
