"""i18n 国际化 - 中/英/日"""
import threading
from typing import Dict, Optional

# 翻译字典
TRANSLATIONS: Dict[str, Dict[str, str]] = {
    "zh-CN": {
        "welcome": "欢迎使用 PackCV",
        "extract": "信息提取",
        "scenarios": "场景",
        "tenant": "租户",
        "billing": "计费",
        "webhook": "Webhook",
        "error.unauthorized": "未授权",
        "error.rate_limited": "请求过于频繁",
        "error.quota_exceeded": "配额已用尽",
        "success": "操作成功",
        "scenarios.license": "营业执照",
        "scenarios.invoice": "发票",
        "scenarios.contract": "合同",
        "scenarios.id_card": "身份证",
    },
    "en": {
        "welcome": "Welcome to PackCV",
        "extract": "Extract",
        "scenarios": "Scenarios",
        "tenant": "Tenant",
        "billing": "Billing",
        "webhook": "Webhook",
        "error.unauthorized": "Unauthorized",
        "error.rate_limited": "Rate limited",
        "error.quota_exceeded": "Quota exceeded",
        "success": "Success",
        "scenarios.license": "Business License",
        "scenarios.invoice": "Invoice",
        "scenarios.contract": "Contract",
        "scenarios.id_card": "ID Card",
    },
    "ja": {
        "welcome": "PackCV へようこそ",
        "extract": "情報抽出",
        "scenarios": "シナリオ",
        "tenant": "テナント",
        "billing": "請求",
        "webhook": "Webhook",
        "error.unauthorized": "認証されていません",
        "error.rate_limited": "レート制限",
        "error.quota_exceeded": "クォータ超過",
        "success": "成功",
        "scenarios.license": "営業許可証",
        "scenarios.invoice": "請求書",
        "scenarios.contract": "契約",
        "scenarios.id_card": "身分証",
    },
}

SUPPORTED_LOCALES = list(TRANSLATIONS.keys())
DEFAULT_LOCALE = "zh-CN"


class Translator:
    """线程安全翻译器"""

    def __init__(self):
        self._local = threading.local()
        self._lock = threading.Lock()
        self._global_locale = DEFAULT_LOCALE

    def get_locale(self) -> str:
        """获取当前 locale"""
        return getattr(self._local, "locale", self._global_locale)

    def set_locale(self, locale: str) -> None:
        """设置当前 locale"""
        if locale in TRANSLATIONS:
            self._local.locale = locale

    def set_global_locale(self, locale: str) -> None:
        """设置全局 locale"""
        with self._lock:
            if locale in TRANSLATIONS:
                self._global_locale = locale

    def translate(self, key: str, locale: Optional[str] = None) -> str:
        """翻译 key"""
        loc = locale or self.get_locale()
        return TRANSLATIONS.get(loc, TRANSLATIONS[DEFAULT_LOCALE]).get(
            key, f"[{key}]"
        )


# 单例
_translator = Translator()


def _(key: str, locale: Optional[str] = None) -> str:
    """翻译快捷函数"""
    return _translator.translate(key, locale)


def set_locale(locale: str) -> None:
    _translator.set_locale(locale)


def set_global_locale(locale: str) -> None:
    _translator.set_global_locale(locale)


def get_locale() -> str:
    return _translator.get_locale()
