"""
PackCV-OCR Python SDK
=====================

官方 Python 客户端,3 行代码接入 OCR 服务。

安装:
    pip install packcv-ocr

快速开始:
    >>> from packcv import PackCVClient
    >>> client = PackCVClient(api_key="pck_live_xxx")
    >>> result = client.extract(image="receipt.jpg", scenario="finance_receipt")
    >>> print(result.fields)

高级用法:
    >>> # 批量处理
    >>> results = client.batch_extract(images=["r1.jpg", "r2.jpg"], scenario="finance_receipt")
    >>>
    >>> # 异步回调 (webhook)
    >>> task = client.extract_async(image="doc.pdf", scenario="contract", webhook="https://...")
    >>> task.poll()  # 轮询获取结果
    >>>
    >>> # 场景自适应
    >>> result = client.extract(image="unknown.jpg")  # 自动识别场景
"""

from .client import PackCVClient, AsyncPackCVClient
from .exceptions import (
    PackCVError,
    AuthenticationError,
    RateLimitError,
    QuotaExceededError,
    ValidationError,
    ServerError,
)
from .models import ExtractResult, ExtractRequest, Scenario, EngineTier

__version__ = "6.3.0"
__all__ = [
    "PackCVClient",
    "AsyncPackCVClient",
    "PackCVError",
    "AuthenticationError",
    "RateLimitError",
    "QuotaExceededError",
    "ValidationError",
    "ServerError",
    "ExtractResult",
    "ExtractRequest",
    "Scenario",
    "EngineTier",
    "__version__",
]
