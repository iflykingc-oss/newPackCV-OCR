"""
Webhook 仓储

使用 Redis 作为持久化层，支持多租户命名空间隔离。
"""
import json
import time
from datetime import datetime
from typing import List, Optional, Dict, Any

from utils.redis_client import redis_client
from webhook.types import (
    WebhookSubscription,
    WebhookEvent,
    DeliveryRecord,
    DLQRecord,
    DeliveryStatus,
)


def _k(prefix: str, *parts: str) -> str:
    return ":".join((prefix,) + parts)


class SubscriptionRepository:
    """订阅仓储

    Key 设计:
    - sub:{subscription_id}         -> WebhookSubscription (JSON)
    - sub:tenant:{tenant_id}        -> Set<subscription_id>
    """

    PREFIX = "webhook:sub"

    def save(self, sub: WebhookSubscription) -> WebhookSubscription:
        redis_client.client.set(
            _k(self.PREFIX, sub.subscription_id),
            sub.model_dump_json(),
        )
        redis_client.client.sadd(
            _k(self.PREFIX, "tenant", sub.tenant_id),
            sub.subscription_id,
        )
        return sub

    def get(self, subscription_id: str) -> Optional[WebhookSubscription]:
        raw = redis_client.client.get(_k(self.PREFIX, subscription_id))
        if not raw:
            return None
        return WebhookSubscription.model_validate_json(raw)

    def list_by_tenant(self, tenant_id: str) -> List[WebhookSubscription]:
        ids = redis_client.client.smembers(_k(self.PREFIX, "tenant", tenant_id)) or set()
        out: List[WebhookSubscription] = []
        for sid in ids:
            sub = self.get(sid)
            if sub is not None:
                out.append(sub)
        return out

    def delete(self, subscription_id: str, tenant_id: str) -> bool:
        removed = redis_client.client.delete(_k(self.PREFIX, subscription_id))
        redis_client.client.srem(_k(self.PREFIX, "tenant", tenant_id), subscription_id)
        return bool(removed)

    def list_all(self) -> List[WebhookSubscription]:
        """列出全部订阅（管理后台用）"""
        # 用 SCAN 避免 KEYS 阻塞
        out: List[WebhookSubscription] = []
        cursor = 0
        while True:
            cursor, keys = redis_client.client.scan(cursor=cursor, match=f"{self.PREFIX}:sub_*", count=100)
            for k in keys:
                raw = redis_client.client.get(k)
                if raw:
                    try:
                        out.append(WebhookSubscription.model_validate_json(raw))
                    except Exception:
                        pass
            if cursor == 0:
                break
        return out


class DeliveryRepository:
    """投递记录仓储

    Key 设计:
    - dlv:{delivery_id}              -> DeliveryRecord (JSON)
    - dlv:event:{event_id}           -> List<delivery_id>
    - dlv:stats:tenant:{tenant_id}   -> Hash {total, success, failed, dlq, p95_ms}
    """

    PREFIX = "webhook:dlv"
    STATS_TTL = 30 * 24 * 3600  # 30 天

    def save(self, record: DeliveryRecord) -> None:
        redis_client.client.set(
            _k(self.PREFIX, record.delivery_id),
            record.model_dump_json(),
            ex=self.STATS_TTL,
        )
        redis_client.client.lpush(
            _k(self.PREFIX, "event", record.event_id),
            record.delivery_id,
        )
        redis_client.client.ltrim(
            _k(self.PREFIX, "event", record.event_id),
            0, 99,  # 保留最近 100 条
        )
        self._update_stats(record)

    def get(self, delivery_id: str) -> Optional[DeliveryRecord]:
        raw = redis_client.client.get(_k(self.PREFIX, delivery_id))
        if not raw:
            return None
        return DeliveryRecord.model_validate_json(raw)

    def list_by_event(self, event_id: str, limit: int = 50) -> List[DeliveryRecord]:
        ids = redis_client.client.lrange(_k(self.PREFIX, "event", event_id), 0, limit - 1) or []
        out: List[DeliveryRecord] = []
        for did in ids:
            rec = self.get(did)
            if rec is not None:
                out.append(rec)
        return out

    def get_stats(self, tenant_id: str) -> Dict[str, Any]:
        key = _k(self.PREFIX, "stats", "tenant", tenant_id)
        stats = redis_client.client.hgetall(key) or {}
        return {
            "total": int(stats.get("total", 0)),
            "success": int(stats.get("success", 0)),
            "failed": int(stats.get("failed", 0)),
            "dead_lettered": int(stats.get("dead_lettered", 0)),
            "avg_duration_ms": float(stats.get("avg_duration_ms", 0.0)),
            "last_event_at": stats.get("last_event_at"),
        }

    def _update_stats(self, record: DeliveryRecord) -> None:
        key = _k(self.PREFIX, "stats", "tenant", record.tenant_id)
        redis_client.client.hincrby(key, "total", 1)
        if record.status == DeliveryStatus.SUCCESS:
            redis_client.client.hincrby(key, "success", 1)
        elif record.status == DeliveryStatus.DEAD_LETTERED:
            redis_client.client.hincrby(key, "dead_lettered", 1)
        else:
            redis_client.client.hincrby(key, "failed", 1)
        if record.duration_ms is not None and record.duration_ms > 0:
            redis_client.client.hset(key, "avg_duration_ms", float(record.duration_ms))
        redis_client.client.hset(key, "last_event_at", datetime.utcnow().isoformat() + "Z")
        redis_client.client.expire(key, self.STATS_TTL)


class DLQRepository:
    """死信队列仓储

    Key 设计:
    - dlq:{record_id}                -> DLQRecord (JSON)
    - dlq:list                      -> List<record_id>
    """

    PREFIX = "webhook:dlq"

    def save(self, record: DLQRecord) -> None:
        redis_client.client.set(
            _k(self.PREFIX, record.record_id),
            record.model_dump_json(),
        )
        redis_client.client.lpush(_k(self.PREFIX, "list"), record.record_id)

    def get(self, record_id: str) -> Optional[DLQRecord]:
        raw = redis_client.client.get(_k(self.PREFIX, record_id))
        if not raw:
            return None
        return DLQRecord.model_validate_json(raw)

    def list_all(self, limit: int = 100) -> List[DLQRecord]:
        ids = redis_client.client.lrange(_k(self.PREFIX, "list"), 0, limit - 1) or []
        out: List[DLQRecord] = []
        for rid in ids:
            rec = self.get(rid)
            if rec is not None:
                out.append(rec)
        return out

    def replay(self, record_id: str) -> Optional[WebhookEvent]:
        """从 DLQ 中取出并准备重投"""
        rec = self.get(record_id)
        if not rec or not rec.can_replay:
            return None
        redis_client.client.delete(_k(self.PREFIX, record_id))
        redis_client.lrem(_k(self.PREFIX, "list"), 1, record_id)
        return rec.event

    def count(self) -> int:
        return int(redis_client.client.llen(_k(self.PREFIX, "list")) or 0)
