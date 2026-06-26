"""
PackCV-OCR Python SDK
多租户文档智能提取服务客户端
"""
from packcv.client import PackCVClient
from packcv.async_client import AsyncPackCVClient
from packcv.types import (
    ExtractRequest,
    ExtractResponse,
    QAResponse,
    UsageInfo,
    TenantInfo,
    ScenarioInfo,
)
from packcv.exceptions import (
    PackCVError,
    AuthenticationError,
    RateLimitError,
    QuotaExceededError,
    APIError,
)

__version__ = "1.0.0"
__all__ = [
    "PackCVClient",
    "AsyncPackCVClient",
    "ExtractRequest",
    "ExtractResponse",
    "QAResponse",
    "UsageInfo",
    "TenantInfo",
    "ScenarioInfo",
    "PackCVError",
    "AuthenticationError",
    "RateLimitError",
    "QuotaExceededError",
    "APIError",
]
