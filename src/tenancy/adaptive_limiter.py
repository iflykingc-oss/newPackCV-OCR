"""自适应限流器

根据历史成功/失败率动态调整限流阈值
参考DeepSeek-V4自适应限流最佳实践
"""
import time
import json
import logging
from typing import Dict, Tuple
from collections import deque

from utils.redis_client import redis_client as get_redis_client

logger = logging.getLogger(__name__)


class AdaptiveRateLimiter:
    """自适应限流器

    设计原则：
    1. 滑动窗口（最近N次请求的成功率）
    2. 动态阈值（错误率高时收紧限流）
    3. 智能恢复（连续成功后逐步放宽）
    4. 防止抖动（使用EWMA指数加权移动平均）
    """

    def __init__(self, window_size: int = 100, alpha: float = 0.3):
        """
        Args:
            window_size: 滑动窗口大小
            alpha: EWMA平滑系数（0-1，越大越敏感）
        """
        self.window_size = window_size
        self.alpha = alpha
        self._local_history: Dict[str, deque] = {}

    def _get_redis_key(self, tenant_id: str) -> str:
        return f"adaptive:{tenant_id}"

    async def record_result(
        self,
        tenant_id: str,
        success: bool,
        response_time: float = 0.0,
    ):
        """记录一次请求结果

        Args:
            tenant_id: 租户ID
            success: 是否成功
            response_time: 响应时间（秒）
        """
        redis = get_redis_client
        key = self._get_redis_key(tenant_id)

        result = {
            "success": success,
            "response_time": response_time,
            "timestamp": time.time(),
        }

        try:
            redis = get_redis_client
            raw_redis = redis.client
            # 1. 写入Redis List（最近N次）
            raw_redis.lpush(f"packcv:adaptive:{key}", json.dumps(result))
            raw_redis.ltrim(f"packcv:adaptive:{key}", 0, self.window_size - 1)
            raw_redis.expire(f"packcv:adaptive:{key}", 3600)  # 1小时过期

            # 2. 更新EWMA（指数加权移动平均）
            await self._update_ewma(tenant_id, success, response_time)
        except Exception as e:
            logger.warning(f"Failed to record result: {e}")
            # 降级到本地内存
            self._record_local(tenant_id, result)

    def _record_local(self, tenant_id: str, result: dict):
        """本地内存降级记录"""
        if tenant_id not in self._local_history:
            self._local_history[tenant_id] = deque(maxlen=self.window_size)
        self._local_history[tenant_id].appendleft(result)

    async def _update_ewma(
        self,
        tenant_id: str,
        success: bool,
        response_time: float,
    ):
        """更新EWMA指标"""
        redis = get_redis_client
        ewma_key = f"{self._get_redis_key(tenant_id)}:ewma"

        try:
            # 获取旧值
            old = redis.hgetall("adaptive", ewma_key)
            old_success_rate = float(old.get("success_rate", 1.0))
            old_avg_rt = float(old.get("avg_response_time", 0.0))

            # 计算新值
            new_success_rate = (
                self.alpha * (1.0 if success else 0.0) +
                (1 - self.alpha) * old_success_rate
            )
            new_avg_rt = (
                self.alpha * response_time +
                (1 - self.alpha) * old_avg_rt
            )

            redis.hset("adaptive", ewma_key, mapping={
                "success_rate": str(new_success_rate),
                "avg_response_time": str(new_avg_rt),
                "updated_at": str(time.time()),
            })
            redis.expire("adaptive", ewma_key, 3600)
        except Exception as e:
            logger.warning(f"Failed to update EWMA: {e}")

    async def get_metrics(self, tenant_id: str) -> dict:
        """获取自适应指标"""
        redis = get_redis_client
        ewma_key = f"{self._get_redis_key(tenant_id)}:ewma"

        try:
            data = redis.hgetall("adaptive", ewma_key)
            return {
                "success_rate": float(data.get("success_rate", 1.0)),
                "avg_response_time": float(data.get("avg_response_time", 0.0)),
                "updated_at": float(data.get("updated_at", 0)),
            }
        except Exception:
            return {
                "success_rate": 1.0,
                "avg_response_time": 0.0,
                "updated_at": 0,
            }

    async def adjust_limit(
        self,
        tenant_id: str,
        base_limit: int,
    ) -> int:
        """根据自适应指标调整限流阈值

        Args:
            tenant_id: 租户ID
            base_limit: 基础限流值

        Returns:
            调整后的限流值
        """
        metrics = await self.get_metrics(tenant_id)
        success_rate = metrics["success_rate"]
        avg_rt = metrics["avg_response_time"]

        # 1. 根据成功率调整
        if success_rate >= 0.95:
            # 高成功率：放宽限流（最多+20%）
            multiplier = 1.0 + min(0.2, (success_rate - 0.95) * 4)
        elif success_rate >= 0.8:
            # 中等成功率：维持
            multiplier = 1.0
        elif success_rate >= 0.5:
            # 低成功率：收紧（-30%）
            multiplier = 0.7
        else:
            # 极低成功率：严厉收紧（-50%）
            multiplier = 0.5

        # 2. 根据响应时间微调
        if avg_rt > 10.0:
            # 平均响应时间>10s：再收紧20%
            multiplier *= 0.8
        elif avg_rt > 5.0:
            multiplier *= 0.9

        adjusted = int(base_limit * multiplier)
        logger.debug(
            f"Adaptive limit for {tenant_id}: "
            f"success_rate={success_rate:.2%}, avg_rt={avg_rt:.2f}s, "
            f"base={base_limit}, adjusted={adjusted}"
        )
        return adjusted
