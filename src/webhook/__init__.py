"""
Webhook 生态模块

提供完整的事件订阅生态：
- 订阅管理
- 签名验证
- 重试机制（指数退避）
- 死信队列（DLQ）
- 投递统计
"""
from webhook.signing import WebhookSigner, verify_signature
from webhook.delivery.retry import RetryPolicy, ExponentialBackoff
from webhook.delivery.dispatcher import EventDispatcher, get_dispatcher
from webhook.storage.repository import (
    SubscriptionRepository,
    DeliveryRepository,
    DLQRepository,
)
from webhook.types import (
    WebhookEvent,
    WebhookSubscription,
    DeliveryRecord,
    DLQRecord,
    EventType,
    DeliveryStatus,
)

__version__ = "1.0.0"

__all__ = [
    "WebhookSigner",
    "verify_signature",
    "RetryPolicy",
    "ExponentialBackoff",
    "EventDispatcher",
    "get_dispatcher",
    "SubscriptionRepository",
    "DeliveryRepository",
    "DLQRepository",
    "WebhookEvent",
    "WebhookSubscription",
    "DeliveryRecord",
    "DLQRecord",
    "EventType",
    "DeliveryStatus",
    "__version__",
]
