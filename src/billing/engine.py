"""计费引擎

支持4种计费模式：
1. by_token: 按Token数计费
2. by_call: 按调用次数计费
3. package: 套餐制
4. hybrid: 混合（套餐+超额按量，推荐）
"""
import time
import json
import logging
from typing import Optional
from datetime import datetime, timezone
from enum import Enum
from pydantic import BaseModel, Field

from utils.redis_client import redis_client as get_redis_client

logger = logging.getLogger(__name__)


class BillingMode(str, Enum):
    """计费模式"""
    BY_TOKEN = "by_token"
    BY_CALL = "by_call"
    PACKAGE = "package"
    HYBRID = "hybrid"


# 模型Token单价（美元/1K tokens，2026年豆包2.0系列参考价）
MODEL_PRICING = {
    "doubao-seed-2-0-pro-260215": {"input": 0.0008, "output": 0.002},
    "doubao-seed-2-0-lite-260215": {"input": 0.0003, "output": 0.0006},
    "doubao-seed-2-0-mini-260215": {"input": 0.0001, "output": 0.0002},
    "kimi-k2-5-260127": {"input": 0.001, "output": 0.002},
    "qwen-3-5-plus-260215": {"input": 0.0008, "output": 0.002},
}

# 套餐价格（人民币元/月）
PACKAGE_PRICING = {
    "free": {"monthly_fee": 0, "monthly_quota": 1000, "overage_rate": 0.0005},
    "basic": {"monthly_fee": 99, "monthly_quota": 5000, "overage_rate": 0.0003},
    "pro": {"monthly_fee": 999, "monthly_quota": 50000, "overage_rate": 0.0002},
    "enterprise": {"monthly_fee": 9999, "monthly_quota": 500000, "overage_rate": 0.0001},
    "flagship": {"monthly_fee": 99999, "monthly_quota": 5000000, "overage_rate": 0.00005},
}

# 汇率（USD -> CNY）
USD_TO_CNY = 7.2


class UsageRecord(BaseModel):
    """使用记录"""
    tenant_id: str
    request_id: str
    timestamp: float
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    scenario: Optional[str] = None
    latency_ms: float = 0.0
    success: bool = True


class BillingEngine:
    """计费引擎"""

    def __init__(self):
        self.redis = None  # 懒加载

    def _get_redis(self):
        if self.redis is None:
            self.redis = get_redis_client
        return self.redis

    def calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> dict:
        """计算单次调用成本（美元）"""
        pricing = MODEL_PRICING.get(model, {"input": 0, "output": 0})
        cost_usd = (
            input_tokens * pricing["input"] / 1000 +
            output_tokens * pricing["output"] / 1000
        )
        return {
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cost_usd": round(cost_usd, 6),
            "cost_cny": round(cost_usd * USD_TO_CNY, 4),
        }

    async def record_usage(
        self,
        tenant_id: str,
        request_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        scenario: Optional[str] = None,
        latency_ms: float = 0.0,
        success: bool = True,
    ) -> dict:
        """记录使用量

        写入Redis：
        - Hash usage:{tenant_id}:{YYYYMM}（月度聚合）
        - List usage:{tenant_id}:detail（详细记录）
        """
        cost = self.calculate_cost(model, input_tokens, output_tokens)
        year_month = datetime.now(timezone.utc).strftime("%Y%m")

        record = UsageRecord(
            tenant_id=tenant_id,
            request_id=request_id,
            timestamp=time.time(),
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost["cost_usd"],
            scenario=scenario,
            latency_ms=latency_ms,
            success=success,
        )

        try:
            redis = self._get_redis()
            raw_redis = redis.client
            monthly_key = f"billing:usage:{tenant_id}:{year_month}"
            detail_key = f"billing:detail:{tenant_id}"
            model_key = f"billing:model:{tenant_id}:{model}:{year_month}"

            # 1. 月度聚合（用 raw_redis 直接操作，避免命名空间差异）
            raw_redis.hincrby(monthly_key, "total_tokens", input_tokens + output_tokens)
            raw_redis.hincrby(monthly_key, "input_tokens", input_tokens)
            raw_redis.hincrby(monthly_key, "output_tokens", output_tokens)
            raw_redis.hincrby(monthly_key, "call_count", 1)
            if not success:
                raw_redis.hincrby(monthly_key, "failed_count", 1)
            raw_redis.hincrbyfloat(monthly_key, "cost_usd", cost["cost_usd"])
            raw_redis.expire(monthly_key, 90 * 86400)

            # 2. 详细记录（最近1000条）
            raw_redis.lpush(detail_key, record.model_dump_json())
            raw_redis.ltrim(detail_key, 0, 999)
            raw_redis.expire(detail_key, 30 * 86400)

            # 3. 按模型聚合
            raw_redis.hincrby(model_key, "tokens", input_tokens + output_tokens)
            raw_redis.hincrby(model_key, "count", 1)
            raw_redis.hincrbyfloat(model_key, "cost", cost["cost_usd"])
            raw_redis.expire(model_key, 90 * 86400)

            logger.debug(
                f"Recorded usage: tenant={tenant_id}, model={model}, "
                f"tokens={input_tokens+output_tokens}, cost=${cost['cost_usd']:.6f}"
            )

        except Exception as e:
            logger.warning(f"Failed to record usage: {e}")

        return cost

    async def get_monthly_usage(self, tenant_id: str, year_month: Optional[str] = None) -> dict:
        """获取月度使用统计"""
        if year_month is None:
            year_month = datetime.now(timezone.utc).strftime("%Y%m")

        try:
            redis = self._get_redis()
            raw_redis = redis.client
            monthly_key = f"billing:usage:{tenant_id}:{year_month}"
            data = raw_redis.hgetall(monthly_key)

            # 兼容 bytes/str
            def _get(k, default=0):
                if k in data:
                    v = data[k]
                elif k.encode() in data:
                    v = data[k.encode()]
                else:
                    return default
                if isinstance(v, bytes):
                    v = v.decode()
                return v

            return {
                "year_month": year_month,
                "total_tokens": int(_get("total_tokens", 0)),
                "input_tokens": int(_get("input_tokens", 0)),
                "output_tokens": int(_get("output_tokens", 0)),
                "call_count": int(_get("call_count", 0)),
                "failed_count": int(_get("failed_count", 0)),
                "cost_usd": float(_get("cost_usd", 0.0)),
                "cost_cny": round(float(_get("cost_usd", 0.0)) * USD_TO_CNY, 4),
            }
        except Exception as e:
            logger.warning(f"Failed to get usage: {e}")
            return {
                "year_month": year_month,
                "total_tokens": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "call_count": 0,
                "failed_count": 0,
                "cost_usd": 0.0,
                "cost_cny": 0.0,
            }

    async def get_recent_records(self, tenant_id: str, limit: int = 100) -> list:
        """获取最近使用记录"""
        try:
            redis = self._get_redis()
            raw_redis = redis.client
            detail_key = f"billing:detail:{tenant_id}"
            raw = raw_redis.lrange(detail_key, 0, limit - 1)
            return [json.loads(r) for r in raw]
        except Exception as e:
            logger.warning(f"Failed to get records: {e}")
            return []


# 全局计费引擎实例
billing_engine = BillingEngine()
