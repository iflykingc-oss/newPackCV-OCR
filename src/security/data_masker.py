"""数据脱敏器

合规要求：
- 身份证/手机/银行卡/邮箱等敏感信息必须脱敏
- 支持完全脱敏和部分脱敏
- 支持结构化数据递归脱敏
"""
import re
import logging
from typing import Optional, Literal

logger = logging.getLogger(__name__)


class DataMasker:
    """数据脱敏器"""

    # 敏感信息正则表达式
    SENSITIVE_PATTERNS = {
        "id_card": r'[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]',
        "phone_cn": r'(?<!\d)1[3-9]\d{9}(?!\d)',
        "phone_global": r'(?<!\d)\+?[1-9]\d{9,14}(?!\d)',
        "bank_card": r'(?<!\d)\d{16,19}(?!\d)',
        "email": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        "credit_card": r'(?<!\d)(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})(?!\d)',
        "ipv4": r'(?<!\d)(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(?!\d)',
        "api_key": r'(?i)(?:api[_-]?key|apikey|token|secret)["\']?\s*[:=]\s*["\']?([a-zA-Z0-9_\-]{16,})',
    }

    # 敏感字段名（自动检测）
    SENSITIVE_KEYS = {
        "id_card", "身份证", "身份证号", "idcard", "id_number", "ssn",
        "phone", "手机", "手机号", "mobile", "telephone", "tel",
        "bank_card", "银行卡", "卡号", "bankaccount", "account_number",
        "email", "邮箱", "邮件",
        "name", "姓名", "real_name", "fullname",
        "address", "地址", "住址", "家庭住址",
        "password", "密码", "pwd", "passwd",
        "credit_card", "信用卡",
        "secret", "密钥", "token", "apikey", "api_key",
    }

    @classmethod
    def mask_text(
        cls,
        text: str,
        mask_type: Literal["full", "partial", "hash"] = "partial",
        patterns: Optional[list] = None,
    ) -> str:
        """脱敏文本

        Args:
            text: 输入文本
            mask_type: full=完全脱敏, partial=保留首尾, hash=哈希替换
            patterns: 要匹配的正则列表，默认全部
        """
        if not text or not isinstance(text, str):
            return text

        if patterns is None:
            patterns = list(cls.SENSITIVE_PATTERNS.keys())

        masked = text
        for pattern_name in patterns:
            pattern = cls.SENSITIVE_PATTERNS.get(pattern_name)
            if not pattern:
                continue

            if mask_type == "full":
                masked = re.sub(
                    pattern,
                    f"***{pattern_name}已脱敏***",
                    masked,
                    flags=re.IGNORECASE,
                )
            elif mask_type == "partial":
                def partial_replace(match):
                    s = match.group(0)
                    if len(s) <= 2:
                        return "*" * len(s)
                    elif len(s) <= 6:
                        return s[0] + "*" * (len(s) - 2) + s[-1]
                    else:
                        return s[:2] + "*" * (len(s) - 4) + s[-2:]
                masked = re.sub(pattern, partial_replace, masked, flags=re.IGNORECASE)
            elif mask_type == "hash":
                import hashlib
                def hash_replace(match):
                    s = match.group(0)
                    h = hashlib.md5(s.encode()).hexdigest()[:8]
                    return f"***{h}***"
                masked = re.sub(pattern, hash_replace, masked, flags=re.IGNORECASE)

        return masked

    @classmethod
    def mask_dict(
        cls,
        data: dict,
        mask_type: Literal["full", "partial", "hash"] = "partial",
        deep: bool = True,
    ) -> dict:
        """脱敏字典数据"""
        if not isinstance(data, dict):
            return data

        masked = {}
        for key, value in data.items():
            key_lower = str(key).lower()
            is_sensitive_key = any(
                sensitive in key_lower
                for sensitive in cls.SENSITIVE_KEYS
            )

            if isinstance(value, str) and (is_sensitive_key or cls._contains_sensitive(value)):
                masked[key] = cls.mask_text(value, mask_type)
            elif isinstance(value, dict) and deep:
                masked[key] = cls.mask_dict(value, mask_type, deep)
            elif isinstance(value, list) and deep:
                masked[key] = [
                    cls.mask_dict(item, mask_type, deep) if isinstance(item, dict)
                    else cls.mask_text(item, mask_type) if isinstance(item, str)
                    else item
                    for item in value
                ]
            else:
                masked[key] = value

        return masked

    @classmethod
    def _contains_sensitive(cls, text: str) -> bool:
        """检查文本是否包含敏感信息"""
        if not text:
            return False
        for pattern in cls.SENSITIVE_PATTERNS.values():
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    @classmethod
    def detect_sensitive(cls, text: str) -> list:
        """检测文本中包含的敏感信息类型"""
        if not text:
            return []
        detected = []
        for name, pattern in cls.SENSITIVE_PATTERNS.items():
            if re.search(pattern, text, re.IGNORECASE):
                detected.append(name)
        return detected

    @classmethod
    def validate_safe(cls, data: dict) -> tuple:
        """验证数据是否安全（无敏感信息泄露）

        Returns:
            (is_safe, issues) 元组
        """
        issues = []
        if not isinstance(data, dict):
            return True, []

        def _check(obj, path="root"):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    _check(v, f"{path}.{k}")
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    _check(item, f"{path}[{i}]")
            elif isinstance(obj, str):
                detected = cls.detect_sensitive(obj)
                if detected:
                    issues.append({
                        "path": path,
                        "types": detected,
                        "preview": obj[:50] + "..." if len(obj) > 50 else obj,
                    })

        _check(data)
        return len(issues) == 0, issues
