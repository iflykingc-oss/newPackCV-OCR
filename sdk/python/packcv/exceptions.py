"""PackCV-OCR 异常体系"""
from typing import Any, Dict, Optional


class PackCVError(Exception):
    """基础异常"""

    def __init__(self, message: str, status_code: Optional[int] = None, response: Optional[Dict[str, Any]] = None):
        self.message = message
        self.status_code = status_code
        self.response = response or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.status_code:
            return f"[{self.status_code}] {self.message}"
        return self.message


class AuthenticationError(PackCVError):
    """鉴权失败 (401)"""


class RateLimitError(PackCVError):
    """触发限流 (429)"""

    def __init__(self, message: str, retry_after: Optional[int] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class QuotaExceededError(PackCVError):
    """配额超限 (402)"""


class APIError(PackCVError):
    """通用API错误 (4xx/5xx)"""
