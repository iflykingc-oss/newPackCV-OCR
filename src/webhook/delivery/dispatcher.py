"""
Webhook 事件分发器

负责将事件投递到所有匹配的订阅方，支持：
- 同步/异步投递
- 自动重试（指数退避）
- 死信队列
- 投递统计
"""
import json
import time
import logging
import threading
from typing import List, Dict, Any, Optional

import httpx

from webhook.types import (
    WebhookEvent,
    WebhookSubscription,
    DeliveryRecord,
    DLQRecord,
    DeliveryStatus,
)
from webhook.signing import WebhookSigner
from webhook.delivery.retry import RetryPolicy, ExponentialBackoff
from webhook.storage.repository import (
    SubscriptionRepository,
    DeliveryRepository,
    DLQRepository,
)

logger = logging.getLogger("webhook.dispatcher")


class EventDispatcher:
    """事件分发器

    使用方式:
        dispatcher = EventDispatcher()
        dispatcher.dispatch(event)  # 同步版
        dispatcher.dispatch_async(event)  # 异步版（后台线程）
    """

    def __init__(
        self,
        sub_repo: Optional[SubscriptionRepository] = None,
        dlv_repo: Optional[DeliveryRepository] = None,
        dlq_repo: Optional[DLQRepository] = None,
        retry_policy: Optional[RetryPolicy] = None,
        http_timeout: float = 10.0,
    ) -> None:
        self.sub_repo = sub_repo or SubscriptionRepository()
        self.dlv_repo = dlv_repo or DeliveryRepository()
        self.dlq_repo = dlq_repo or DLQRepository()
        self.retry_policy = retry_policy or RetryPolicy()
        self.http_timeout = http_timeout

    def dispatch(self, event: WebhookEvent) -> List[DeliveryRecord]:
        """同步分发事件到所有匹配的订阅方"""
        subs = self._find_matching_subs(event)
        records: List[DeliveryRecord] = []
        for sub in subs:
            record = self._deliver_with_retry(event, sub)
            records.append(record)
        return records

    def dispatch_async(self, event: WebhookEvent) -> threading.Thread:
        """异步分发（后台线程）"""
        t = threading.Thread(
            target=self.dispatch,
            args=(event,),
            daemon=True,
            name=f"webhook-dispatch-{event.event_id[:8]}",
        )
        t.start()
        return t

    def _find_matching_subs(self, event: WebhookEvent) -> List[WebhookSubscription]:
        subs = self.sub_repo.list_by_tenant(event.tenant_id)
        return [s for s in subs if s.active and event.event_type in s.events]

    def _deliver_with_retry(
        self,
        event: WebhookEvent,
        sub: WebhookSubscription,
    ) -> DeliveryRecord:
        """投递单个订阅，包含完整重试逻辑"""
        payload = json.dumps(event.to_dict(), separators=(",", ":")).encode("utf-8")
        signer = WebhookSigner(sub.secret)
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "PackCV-Webhook/1.0",
            "X-PackCV-Event-Id": event.event_id,
            "X-PackCV-Event-Type": event.event_type.value,
            "X-PackCV-Subscription-Id": sub.subscription_id,
        }
        if event.idempotency_key:
            headers["X-PackCV-Idempotency-Key"] = event.idempotency_key

        backoff = ExponentialBackoff(
            base=self.retry_policy.base_delay,
            max_delay=self.retry_policy.max_delay,
            multiplier=self.retry_policy.multiplier,
        )

        attempt = 0
        last_record: Optional[DeliveryRecord] = None
        max_attempts = self.retry_policy.max_attempts

        while attempt < max_attempts:
            attempt += 1
            # 每次重试都重新签名
            headers["X-PackCV-Signature"] = signer.sign(payload)
            headers["X-PackCV-Attempt"] = str(attempt)
            headers["X-PackCV-Delivery-Id"] = (
                f"dlv_{event.event_id[:8]}_{attempt}"
            )

            start = time.time()
            record = self._attempt_delivery(event, sub, payload, headers, attempt)
            record.duration_ms = int((time.time() - start) * 1000)
            last_record = record

            if record.status == DeliveryStatus.SUCCESS:
                self.dlv_repo.save(record)
                logger.info(
                    "webhook delivered event=%s sub=%s attempt=%d code=%d duration=%dms",
                    event.event_id, sub.subscription_id, attempt,
                    record.response_code or 0, record.duration_ms,
                )
                return record

            # 记录失败
            self.dlv_repo.save(record)
            logger.warning(
                "webhook failed event=%s sub=%s attempt=%d code=%d err=%s",
                event.event_id, sub.subscription_id, attempt,
                record.response_code or 0, record.error or "unknown",
            )

            # 如果还有重试机会，等待退避
            if attempt < max_attempts:
                delay = backoff.compute(attempt)
                time.sleep(delay)

        # 全部重试失败 -> DLQ
        if last_record is not None:
            dlq = DLQRecord(
                event=event,
                subscription_id=sub.subscription_id,
                total_attempts=attempt,
                last_error=last_record.error or "exhausted",
                last_response_code=last_record.response_code,
            )
            self.dlq_repo.save(dlq)
            # 同时更新最后一条记录为 DEAD_LETTERED
            last_record.status = DeliveryStatus.DEAD_LETTERED
            self.dlv_repo.save(last_record)
            logger.error(
                "webhook dead-lettered event=%s sub=%s attempts=%d",
                event.event_id, sub.subscription_id, attempt,
            )

        return last_record  # type: ignore[return-value]

    def _attempt_delivery(
        self,
        event: WebhookEvent,
        sub: WebhookSubscription,
        payload: bytes,
        headers: Dict[str, str],
        attempt: int,
    ) -> DeliveryRecord:
        """单次投递尝试"""
        record = DeliveryRecord(
            event_id=event.event_id,
            subscription_id=sub.subscription_id,
            tenant_id=event.tenant_id,
            url=str(sub.url),
            attempt=attempt,
        )
        try:
            with httpx.Client(timeout=self.http_timeout) as client:
                resp = client.post(str(sub.url), content=payload, headers=headers)
            record.response_code = resp.status_code
            record.response_body = resp.text[:1000]  # 截断
            if 200 <= resp.status_code < 300:
                record.status = DeliveryStatus.SUCCESS
            else:
                record.status = DeliveryStatus.FAILED
                record.error = f"HTTP {resp.status_code}"
        except httpx.TimeoutException:
            record.status = DeliveryStatus.FAILED
            record.error = "timeout"
        except httpx.RequestError as e:
            record.status = DeliveryStatus.FAILED
            record.error = f"connection_error: {type(e).__name__}"
        except Exception as e:
            record.status = DeliveryStatus.FAILED
            record.error = f"unexpected: {type(e).__name__}: {e}"
        finally:
            record.completed_at = (
                __import__("datetime").datetime.utcnow().isoformat() + "Z"
            )
        return record


# 全局单例
_default_dispatcher: Optional[EventDispatcher] = None


def get_dispatcher() -> EventDispatcher:
    global _default_dispatcher
    if _default_dispatcher is None:
        _default_dispatcher = EventDispatcher()
    return _default_dispatcher
