# -*- coding: utf-8 -*-
"""
IM Platform Dispatcher - 命令路由 + 多平台统一管理
- 注册所有平台adapter
- 命令路由（/ocr /extract /help /ping）
- 跨平台消息构造
"""

import json
import re
import logging
import time
from typing import Any, Dict, List, Optional

from .base import IMPlatform, IMMessageType
from .feishu_bot import FeishuPlatform
from .dingtalk_bot import DingTalkPlatform
from .wecom_bot import WeComPlatform

logger = logging.getLogger(__name__)


# 命令路由表：命令名 → 描述
COMMANDS: Dict[str, Dict[str, str]] = {
    "/ocr": {"desc": "上传图片识别包装信息，例：/ocr <图片>", "handler": "ocr"},
    "/extract": {"desc": "提取结构化字段，例：/extract <图片>", "handler": "extract"},
    "/help": {"desc": "查看帮助", "handler": "help"},
    "/ping": {"desc": "健康检查", "handler": "ping"},
    "/stats": {"desc": "查看调用统计", "handler": "stats"},
}


class IMPlatformDispatcher:
    """多平台统一调度器"""

    def __init__(self):
        self._adapters: Dict[str, IMPlatform] = {}
        self.register_default_adapters()

    def register_adapter(self, platform: str, adapter: IMPlatform) -> None:
        self._adapters[platform] = adapter
        logger.info(f"已注册IM平台: {platform}")

    def register_default_adapters(self) -> None:
        self.register_adapter("feishu", FeishuPlatform())
        self.register_adapter("dingtalk", DingTalkPlatform())
        self.register_adapter("wecom", WeComPlatform())

    def get_adapter(self, platform: str) -> Optional[IMPlatform]:
        return self._adapters.get(platform)

    def parse_command(self, text: str) -> Dict[str, Any]:
        """解析用户命令
        Returns:
            {"command": str, "args": str, "is_command": bool}
        """
        text = (text or "").strip()
        if not text.startswith("/"):
            return {"command": "", "args": text, "is_command": False}
        # 提取命令
        parts = text.split(None, 1)
        cmd = parts[0].lower().split("@")[0]  # 去除@bot
        args = parts[1] if len(parts) > 1 else ""
        return {
            "command": cmd,
            "args": args.strip(),
            "is_command": True,
            "known": cmd in COMMANDS
        }

    def route_command(self, command: str, args: str, image_url: str = "") -> Dict[str, Any]:
        """根据命令路由到对应handler，返回响应payload（统一格式）"""
        if command == "/help" or not command:
            return self._build_help_response()
        if command == "/ping":
            return {
                "title": "🏓 Pong",
                "content": f"服务正常 | 时间 {time.strftime('%Y-%m-%d %H:%M:%S')}",
                "actions": []
            }
        if command == "/stats":
            return {
                "title": "📊 调用统计",
                "content": "请访问 /api/admin/stats 查看完整统计Dashboard",
                "actions": [
                    {"tag": "button", "text": {"content": "查看Dashboard", "tag": "plain_text"},
                     "type": "primary", "url": "/api/admin/stats"}
                ]
            }
        if command in ("/ocr", "/extract"):
            if not image_url:
                return {
                    "title": "⚠️ 请提供图片",
                    "content": "用法：/ocr <图片>  或直接发送图片消息",
                    "actions": []
                }
            # 异步处理（实际由web_server异步调用OCR工作流）
            return {
                "title": "🔍 正在识别",
                "content": f"已收到图片，正在分析：{image_url[:80]}...",
                "actions": []
            }
        return {
            "title": "❓ 未知命令",
            "content": f"未知命令：{command}，输入 /help 查看帮助",
            "actions": []
        }

    def _build_help_response(self) -> Dict[str, Any]:
        lines = ["**可用命令**\n"]
        for cmd, info in COMMANDS.items():
            lines.append(f"- `{cmd}` - {info['desc']}")
        return {
            "title": "📖 PackCV-OCR 帮助",
            "content": "\n".join(lines),
            "actions": []
        }

    def dispatch_to_platform(
        self,
        platform: str,
        title: str,
        content: str,
        actions: Optional[List[Dict[str, Any]]] = None,
        webhook_url: str = "",
        template: str = "blue"
    ) -> Dict[str, Any]:
        """根据平台构造对应格式的卡片消息payload"""
        adapter = self.get_adapter(platform)
        if not adapter:
            return {"error": f"未知平台: {platform}"}
        if platform == "feishu":
            return adapter.build_card_message(title, content, actions, template)
        if platform == "dingtalk":
            return adapter.build_card_message(title, content, actions, template)
        if platform == "wecom":
            return adapter.build_card_message(title, content, actions, template)
        return {"error": f"平台{platform}未实现"}


# 全局单例
_dispatcher: Optional[IMPlatformDispatcher] = None


def get_dispatcher() -> IMPlatformDispatcher:
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = IMPlatformDispatcher()
    return _dispatcher
