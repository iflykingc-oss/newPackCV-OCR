"""审计日志器

全链路操作追溯，符合金融级合规要求
"""
import json
import time
import logging
import asyncio
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field

from utils.redis_client import redis_client as get_redis_client

logger = logging.getLogger(__name__)


class AuditAction(str, Enum):
    """审计动作"""
    # 租户管理
    TENANT_CREATE = "tenant.create"
    TENANT_UPDATE = "tenant.update"
    TENANT_SUSPEND = "tenant.suspend"
    TENANT_DELETE = "tenant.delete"
    # API密钥
    APIKEY_CREATE = "apikey.create"
    APIKEY_REVOKE = "apikey.revoke"
    # 鉴权
    AUTH_SUCCESS = "auth.success"
    AUTH_FAIL = "auth.fail"
    # API调用
    API_CALL = "api.call"
    API_SUCCESS = "api.success"
    API_FAIL = "api.fail"
    # 数据操作
    DATA_READ = "data.read"
    DATA_WRITE = "data.write"
    DATA_DELETE = "data.delete"
    DATA_EXPORT = "data.export"
    # 计费
    BILLING_DEDUCT = "billing.deduct"
    BILLING_REFUND = "billing.refund"
    INVOICE_ISSUE = "invoice.issue"
    INVOICE_PAID = "invoice.paid"
    # 模型
    MODEL_INVOKE = "model.invoke"
    MODEL_FAILOVER = "model.failover"
    MODEL_DEGRADE = "model.degrade"
    # 限流
    RATE_LIMIT_HIT = "ratelimit.hit"
    # 配额
    QUOTA_WARNING = "quota.warning"
    QUOTA_EXCEEDED = "quota.exceeded"


class AuditLogEntry(BaseModel):
    """审计日志条目"""
    log_id: str = Field(default_factory=lambda: f"audit-{int(time.time() * 1000)}-{hash(str(time.time())) % 10000:04d}")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    action: str
    tenant_id: Optional[str] = None
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    resource: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    status: str = "success"  # success/failure/warning
    metadata: dict = Field(default_factory=dict)
    error_message: Optional[str] = None


class AuditLogger:
    """审计日志器

    实现全链路可追溯：
    1. 写入Redis（最近N条）
    2. 写入结构化日志文件（保留180天）
    3. 敏感操作实时告警
    """

    # 敏感操作（需要实时告警）
    SENSITIVE_ACTIONS = {
        AuditAction.TENANT_DELETE,
        AuditAction.APIKEY_REVOKE,
        AuditAction.DATA_DELETE,
        AuditAction.DATA_EXPORT,
        AuditAction.BILLING_REFUND,
    }

    # 失败操作（需要告警）
    FAILURE_ACTIONS = {
        AuditAction.AUTH_FAIL,
        AuditAction.API_FAIL,
        AuditAction.RATE_LIMIT_HIT,
        AuditAction.QUOTA_EXCEEDED,
    }

    def __init__(self):
        self.redis = None
        self._alert_callbacks = []

    def _get_redis(self):
        if self.redis is None:
            self.redis = get_redis_client
        return self.redis

    def register_alert_callback(self, callback):
        """注册告警回调"""
        self._alert_callbacks.append(callback)

    async def log(
        self,
        action: AuditAction,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
        resource: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        status: str = "success",
        metadata: Optional[dict] = None,
        error_message: Optional[str] = None,
    ) -> AuditLogEntry:
        """记录审计日志"""
        entry = AuditLogEntry(
            action=action.value,
            tenant_id=tenant_id,
            user_id=user_id,
            request_id=request_id,
            resource=resource,
            ip_address=ip_address,
            user_agent=user_agent,
            status=status,
            metadata=metadata or {},
            error_message=error_message,
        )

        # 1. 写入Redis（最近10000条）
        try:
            redis = self._get_redis()
            raw_redis = redis.client
            raw_redis.lpush("packcv:audit:audit:recent", entry.model_dump_json())
            raw_redis.ltrim("packcv:audit:audit:recent", 0, 9999)
            raw_redis.expire("packcv:audit:audit:recent", 30 * 86400)

            # 2. 按租户聚合
            if tenant_id:
                tenant_key = f"audit:tenant:{tenant_id}"
                raw_redis.lpush(f"packcv:audit:{tenant_key}", entry.model_dump_json())
                raw_redis.ltrim(f"packcv:audit:{tenant_key}", 0, 999)
                raw_redis.expire(f"packcv:audit:{tenant_key}", 90 * 86400)

        except Exception as e:
            logger.warning(f"Failed to write audit to Redis: {e}")

        # 3. 写入结构化日志
        log_data = entry.model_dump()
        if status == "failure":
            logger.warning(f"AUDIT_FAIL: {json.dumps(log_data, ensure_ascii=False)}")
        elif action in self.SENSITIVE_ACTIONS:
            logger.warning(f"AUDIT_SENSITIVE: {json.dumps(log_data, ensure_ascii=False)}")
        else:
            logger.info(f"AUDIT: {json.dumps(log_data, ensure_ascii=False)}")

        # 4. 触发告警
        if action in self.SENSITIVE_ACTIONS or action in self.FAILURE_ACTIONS:
            await self._trigger_alert(entry)

        return entry

    async def _trigger_alert(self, entry: AuditLogEntry):
        """触发告警"""
        for callback in self._alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(entry)
                else:
                    callback(entry)
            except Exception as e:
                logger.warning(f"Alert callback failed: {e}")

    async def query(
        self,
        tenant_id: Optional[str] = None,
        action: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        limit: int = 100,
    ) -> list:
        """查询审计日志"""
        try:
            redis = self._get_redis()
            raw_redis = redis.client
            key = f"audit:tenant:{tenant_id}" if tenant_id else "audit:recent"
            full_key = f"packcv:audit:{key}"
            raw = raw_redis.lrange(full_key, 0, limit * 2 - 1)

            results = []
            for r in raw:
                try:
                    entry = json.loads(r)
                    if action and entry.get("action") != action:
                        continue
                    if start_time:
                        ts = datetime.fromisoformat(
                            entry["timestamp"].replace("Z", "+00:00")
                        ).timestamp()
                        if ts < start_time:
                            continue
                    if end_time:
                        ts = datetime.fromisoformat(
                            entry["timestamp"].replace("Z", "+00:00")
                        ).timestamp()
                        if ts > end_time:
                            continue
                    results.append(entry)
                    if len(results) >= limit:
                        break
                except Exception:
                    continue
            return results
        except Exception as e:
            logger.warning(f"Failed to query audit: {e}")
            return []


# 全局审计日志器
audit_logger = AuditLogger()
