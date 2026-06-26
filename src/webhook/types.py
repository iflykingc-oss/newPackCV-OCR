"""Webhook 类型定义

提供 WebHook 系统使用的所有数据模型。
"""
from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator, HttpUrl
import uuid


class EventType(str, Enum):
    """支持的事件类型"""
    # 任务相关
    TASK_CREATED = "task.created"
    TASK_STARTED = "task.started"
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"
    # 计费相关
    BILLING_USAGE = "billing.usage"
    BILLING_QUOTA_EXCEEDED = "billing.quota_exceeded"
    BILLING_INVOICE_READY = "billing.invoice_ready"
    # 租户相关
    TENANT_CREATED = "tenant.created"
    TENANT_SUSPENDED = "tenant.suspended"
    APIKEY_ROTATED = "apikey.rotated"
    # 系统相关
    SYSTEM_DEGRADED = "system.degraded"
    SYSTEM_RECOVERED = "system.recovered"


class DeliveryStatus(str, Enum):
    """投递状态"""
    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    RETRYING = "retrying"
    DEAD_LETTERED = "dead_lettered"


class WebhookEvent(BaseModel):
    """Webhook 事件载荷"""
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType
    tenant_id: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    idempotency_key: Optional[str] = None
    data: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode="json")


class WebhookSubscription(BaseModel):
    """Webhook 订阅"""
    subscription_id: str = Field(default_factory=lambda: f"sub_{uuid.uuid4().hex[:16]}")
    tenant_id: str
    url: HttpUrl
    events: List[EventType]
    secret: str = Field(..., min_length=8, max_length=128, description="HMAC 签名密钥")
    active: bool = True
    description: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    @field_validator("events")
    @classmethod
    def _validate_events(cls, v: List[EventType]) -> List[EventType]:
        if not v:
            raise ValueError("至少订阅一个事件类型")
        return v


class DeliveryRecord(BaseModel):
    """投递记录"""
    delivery_id: str = Field(default_factory=lambda: f"dlv_{uuid.uuid4().hex[:16]}")
    event_id: str
    subscription_id: str
    tenant_id: str
    url: str
    attempt: int = 1
    status: DeliveryStatus = DeliveryStatus.PENDING
    response_code: Optional[int] = None
    response_body: Optional[str] = None
    duration_ms: Optional[int] = None
    error: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    completed_at: Optional[str] = None


class DLQRecord(BaseModel):
    """死信队列记录"""
    record_id: str = Field(default_factory=lambda: f"dlq_{uuid.uuid4().hex[:16]}")
    event: WebhookEvent
    subscription_id: str
    total_attempts: int
    last_error: str
    last_response_code: Optional[int] = None
    failed_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    can_replay: bool = True
