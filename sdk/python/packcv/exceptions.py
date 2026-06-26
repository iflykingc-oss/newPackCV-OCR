"""PackCV SDK - 异常定义"""

from typing import Optional, Dict, Any


class PackCVError(Exception):
    """PackCV SDK 基础异常"""
    def __init__(self, message: str, code: Optional[str] = None, status_code: Optional[int] = None, response: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code
        self.response = response or {}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.message!r}, code={self.code!r})"


class AuthenticationError(PackCVError):
    """API Key 无效或缺失 (401)"""
    pass


class RateLimitError(PackCVError):
    """触发限流 (429)"""
    def __init__(self, message: str, retry_after: Optional[int] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class QuotaExceededError(PackCVError):
    """配额耗尽 (402)"""
    def __init__(self, message: str, quota: Optional[int] = None, used: Optional[int] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.quota = quota
        self.used = used


class ValidationError(PackCVError):
    """请求参数错误 (400)"""
    pass


class ServerError(PackCVError):
    """服务端错误 (5xx)"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, **kwargs)
        self.is_retryable = True
