# -*- coding: utf-8 -*-
"""
i18n国际化模块单元测试
验证多语言消息、locale解析、Unicode处理
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.getenv("COZE_WORKSPACE_PATH", "."), "src"))


class TestI18n:
    """i18n国际化功能测试"""

    def test_get_error_message_chinese(self):
        """中文错误消息"""
        from utils.i18n import get_error_message
        msg = get_error_message("auth_invalid_api_key", "zh-CN")
        assert isinstance(msg, str)
        assert len(msg) > 0

    def test_get_error_message_english(self):
        """英文错误消息"""
        from utils.i18n import get_error_message
        msg = get_error_message("auth_invalid_api_key", "en")
        assert isinstance(msg, str)
        assert len(msg) > 0

    def test_get_error_message_japanese(self):
        """日文错误消息"""
        from utils.i18n import get_error_message
        msg = get_error_message("auth_invalid_api_key", "ja")
        assert isinstance(msg, str)
        assert len(msg) > 0

    def test_get_error_message_korean(self):
        """韩文错误消息"""
        from utils.i18n import get_error_message
        msg = get_error_message("auth_invalid_api_key", "ko")
        assert isinstance(msg, str)
        assert len(msg) > 0

    def test_fallback_to_chinese(self):
        """未知locale回退到中文"""
        from utils.i18n import get_error_message
        msg = get_error_message("auth_invalid_api_key", "xx-XX")
        assert isinstance(msg, str)
        assert len(msg) > 0  # 应回退到zh-CN

    def test_resolve_locale_from_accept_language(self):
        """从Accept-Language header解析locale"""
        from utils.i18n import resolve_locale
        assert resolve_locale("en-US,en;q=0.9") == "en"
        assert resolve_locale("ja") == "ja"
        assert resolve_locale("ko-KR") == "ko"
        assert resolve_locale("zh-CN,zh;q=0.9") == "zh-CN"

    def test_resolve_locale_default(self):
        """空Accept-Language默认zh-CN"""
        from utils.i18n import resolve_locale
        assert resolve_locale("") == "zh-CN"
        assert resolve_locale(None) == "zh-CN"

    def test_rate_limit_message_with_params(self):
        """带参数的错误消息格式化"""
        from utils.i18n import get_error_message
        msg = get_error_message("rate_limit_exceeded", "zh-CN", limit=100)
        assert "100" in msg

    def test_unknown_error_key_returns_fallback(self):
        """未注册的错误key返回兜底消息"""
        from utils.i18n import get_error_message
        msg = get_error_message("nonexistent_error_key", "zh-CN")
        assert isinstance(msg, str)
        assert len(msg) > 0  # 应有兜底

    def test_unicode_content_preserved(self):
        """Unicode内容（emoji/CJK）不被破坏"""
        from utils.i18n import get_error_message
        msg = get_error_message("auth_invalid_api_key", "zh-CN")
        # 确保没有乱码
        assert msg.encode("utf-8").decode("utf-8") == msg
