"""
WebHook 端点 - 事件订阅与分发（Phase 3 W4 升级版）

集成 webhook 引擎：
- 自动重试（指数退避）
- 死信队列（DLQ）
- HMAC-SHA256 签名
- 投递统计
"""
import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, HttpUrl

from webhook import (
    WebhookEvent,
    WebhookSubscription,
    EventType,
    EventDispatcher,
    SubscriptionRepository,
    DeliveryRepository,
    DLQRepository,
    get_dispatcher,
)

logger = logging.getLogger("api.webhooks")
router = APIRouter(prefix="/webhooks", tags=["webhooks"])


# ============= Pydantic 模型 =============

class WebhookSubscribeRequest(BaseModel):
    """订阅 WebHook 请求"""
    tenant_id: str = Field(..., min_length=1, description="租户ID")
    url: HttpUrl = Field(..., description="回调 URL")
    events: List[str] = Field(..., min_length=1, description="事件类型列表")
    secret: str = Field(..., min_length=8, max_length=128, description="HMAC 签名密钥")
    description: Optional[str] = None


class WebhookSubscribeResponse(BaseModel):
    """订阅响应"""
    subscription_id: str
    tenant_id: str
    url: str
    events: List[str]
    active: bool
    created_at: str


class WebhookDispatchRequest(BaseModel):
    """事件分发请求"""
    event_type: str = Field(..., description="事件类型")
    tenant_id: str = Field(..., description="租户ID")
    data: Dict[str, Any] = Field(default_factory=dict)
    idempotency_key: Optional[str] = None


class WebhookDispatchResponse(BaseModel):
    """事件分发响应"""
    event_id: str
    event_type: str
    matched_subscriptions: int
    deliveries: List[Dict[str, Any]]


class WebhookStatsResponse(BaseModel):
    """投递统计响应"""
    tenant_id: str
    total: int
    success: int
    failed: int
    dead_lettered: int
    avg_duration_ms: float
    last_event_at: Optional[str]
    success_rate: float


# ============= 端点 =============

@router.post("/subscribe", response_model=WebhookSubscribeResponse)
async def subscribe_webhook(req: WebhookSubscribeRequest):
    """订阅 WebHook（HMAC 密钥加密存储）"""
    # 校验事件类型
    try:
        event_types = [EventType(e) for e in req.events]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"无效事件类型: {e}")

    sub = WebhookSubscription(
        tenant_id=req.tenant_id,
        url=req.url,
        events=event_types,
        secret=req.secret,
        description=req.description,
    )
    repo = SubscriptionRepository()
    repo.save(sub)
    return WebhookSubscribeResponse(
        subscription_id=sub.subscription_id,
        tenant_id=sub.tenant_id,
        url=str(sub.url),
        events=[e.value for e in sub.events],
        active=sub.active,
        created_at=sub.created_at,
    )


@router.get("/list/{tenant_id}")
async def list_webhooks(tenant_id: str):
    """列出某租户的所有订阅（不含密钥）"""
    repo = SubscriptionRepository()
    subs = repo.list_by_tenant(tenant_id)
    return {
        "tenant_id": tenant_id,
        "count": len(subs),
        "subscriptions": [
            {
                "subscription_id": s.subscription_id,
                "url": str(s.url),
                "events": [e.value for e in s.events],
                "active": s.active,
                "description": s.description,
                "created_at": s.created_at,
            }
            for s in subs
        ],
    }


@router.delete("/{tenant_id}/{subscription_id}")
async def unsubscribe_webhook(tenant_id: str, subscription_id: str):
    """取消订阅"""
    repo = SubscriptionRepository()
    if not repo.delete(subscription_id, tenant_id):
        raise HTTPException(status_code=404, detail="订阅不存在")
    return {"deleted": True, "subscription_id": subscription_id}


@router.post("/dispatch", response_model=WebhookDispatchResponse)
async def dispatch_event(req: WebhookDispatchRequest):
    """
    同步分发事件

    将事件投递到所有匹配的订阅方，**包含完整重试**。
    适合低频重要事件（如任务完成、配额超限）。
    """
    try:
        event_type = EventType(req.event_type)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"无效事件类型: {e}")

    event = WebhookEvent(
        event_type=event_type,
        tenant_id=req.tenant_id,
        data=req.data,
        idempotency_key=req.idempotency_key,
    )
    dispatcher = get_dispatcher()
    records = dispatcher.dispatch(event)

    return WebhookDispatchResponse(
        event_id=event.event_id,
        event_type=event.event_type.value,
        matched_subscriptions=len(records),
        deliveries=[
            {
                "delivery_id": r.delivery_id,
                "subscription_id": r.subscription_id,
                "url": r.url,
                "attempt": r.attempt,
                "status": r.status.value,
                "response_code": r.response_code,
                "duration_ms": r.duration_ms,
                "error": r.error,
            }
            for r in records
        ],
    )


@router.post("/dispatch-async")
async def dispatch_event_async(req: WebhookDispatchRequest):
    """异步分发事件（立即返回，后台线程投递）"""
    try:
        event_type = EventType(req.event_type)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"无效事件类型: {e}")

    event = WebhookEvent(
        event_type=event_type,
        tenant_id=req.tenant_id,
        data=req.data,
        idempotency_key=req.idempotency_key,
    )
    dispatcher = get_dispatcher()
    dispatcher.dispatch_async(event)
    return {
        "event_id": event.event_id,
        "event_type": event.event_type.value,
        "status": "queued",
    }


@router.get("/stats/{tenant_id}", response_model=WebhookStatsResponse)
async def get_delivery_stats(tenant_id: str):
    """获取投递统计"""
    repo = DeliveryRepository()
    stats = repo.get_stats(tenant_id)
    total = stats.get("total", 0)
    success = stats.get("success", 0)
    return WebhookStatsResponse(
        tenant_id=tenant_id,
        total=total,
        success=success,
        failed=stats.get("failed", 0),
        dead_lettered=stats.get("dead_lettered", 0),
        avg_duration_ms=stats.get("avg_duration_ms", 0.0),
        last_event_at=stats.get("last_event_at"),
        success_rate=round(success / total, 4) if total > 0 else 0.0,
    )


@router.get("/dlq")
async def list_dlq(limit: int = Query(default=50, ge=1, le=500)):
    """查看死信队列"""
    repo = DLQRepository()
    records = repo.list_all(limit=limit)
    return {
        "count": repo.count(),
        "records": [
            {
                "record_id": r.record_id,
                "subscription_id": r.subscription_id,
                "event_type": r.event.event_type.value,
                "tenant_id": r.event.tenant_id,
                "total_attempts": r.total_attempts,
                "last_error": r.last_error,
                "last_response_code": r.last_response_code,
                "failed_at": r.failed_at,
            }
            for r in records
        ],
    }


@router.post("/dlq/{record_id}/replay")
async def replay_dlq(record_id: str):
    """从死信队列中重投事件"""
    repo = DLQRepository()
    event = repo.replay(record_id)
    if event is None:
        raise HTTPException(status_code=404, detail="记录不存在或不可重投")
    dispatcher = get_dispatcher()
    records = dispatcher.dispatch(event)
    return {
        "event_id": event.event_id,
        "replayed_to": len(records),
        "deliveries": [
            {"status": r.status.value, "response_code": r.response_code}
            for r in records
        ],
    }


@router.get("/event-types")
async def list_event_types():
    """列出所有可订阅事件类型"""
    descriptions = {
        EventType.TASK_CREATED: ("task.created", "任务创建"),
        EventType.TASK_STARTED: ("task.started", "任务开始处理"),
        EventType.TASK_COMPLETED: ("task.completed", "提取任务完成"),
        EventType.TASK_FAILED: ("task.failed", "提取任务失败"),
        EventType.BILLING_USAGE: ("billing.usage", "用量上报（每小时）"),
        EventType.BILLING_QUOTA_EXCEEDED: ("billing.quota_exceeded", "配额超限"),
        EventType.BILLING_INVOICE_READY: ("billing.invoice_ready", "账单生成"),
        EventType.TENANT_CREATED: ("tenant.created", "新租户开通"),
        EventType.TENANT_SUSPENDED: ("tenant.suspended", "租户暂停"),
        EventType.APIKEY_ROTATED: ("apikey.rotated", "API Key 轮换"),
        EventType.SYSTEM_DEGRADED: ("system.degraded", "系统降级"),
        EventType.SYSTEM_RECOVERED: ("system.recovered", "系统恢复"),
    }
    return {
        "count": len(EventType),
        "events": [
            {"type": k.value, "desc": v[1]}
            for k, v in descriptions.items()
        ],
    }
