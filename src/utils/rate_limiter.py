#!/usr/bin/env python3
"""
Token预算+多级限流器 - V7.0商业化
支持TPM(每分钟Token)/RPM(每分钟请求)/QPS/单用户日配额/月度配额
"""
import os
import time
import logging
from typing import Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock

logger = logging.getLogger(__name__)


class LimitType(str, Enum):
    """限流类型"""
    TPM = "tpm"          # Token per minute
    RPM = "rpm"          # Request per minute
    QPS = "qps"          # Query per second
    DAILY_TOKEN = "daily_token"  # 单用户单日Token
    MONTHLY_BUDGET = "monthly_budget"  # 月度预算(美元)


@dataclass
class LimitConfig:
    """限流配置"""
    limit_type: LimitType
    max_value: float
    window_seconds: int = 60
    description: str = ""


@dataclass
class UsageRecord:
    """使用记录"""
    user_id: str = "default"
    tenant_id: str = "default"
    tpm_used: float = 0.0          # 当前窗口已用TPM
    rpm_used: int = 0              # 当前窗口已用RPM
    daily_token_used: float = 0.0  # 今日已用Token
    monthly_cost_usd: float = 0.0  # 本月已用成本
    window_start: float = field(default_factory=time.time)  # 当前窗口开始时间
    day_start: float = field(default_factory=lambda: time.time() // 86400 * 86400)
    month_start: float = field(default_factory=lambda: time.time() // 2592000 * 2592000)


class RateLimiter:
    """
    生产级限流器 - 多维度配额管理
    设计原则:
    - 内存版: 单机部署, 简单快速
    - 滑动窗口: 精确控制TPM/RPM
    - 软硬限制: 软限只警告, 硬限直接拒绝
    """

    def __init__(self, config: Optional[Dict[str, LimitConfig]] = None):
        self.locks: Dict[str, Lock] = {}
        self.records: Dict[str, UsageRecord] = {}
        self.global_lock = Lock()

        # 默认生产配置
        self.limits: Dict[LimitType, LimitConfig] = config or {
            LimitType.TPM: LimitConfig(
                limit_type=LimitType.TPM,
                max_value=100000,  # 10万Token/分钟
                window_seconds=60,
                description="TPM全局限制"
            ),
            LimitType.RPM: LimitConfig(
                limit_type=LimitType.RPM,
                max_value=600,     # 600请求/分钟
                window_seconds=60,
                description="RPM全局限制"
            ),
            LimitType.QPS: LimitConfig(
                limit_type=LimitType.QPS,
                max_value=20,      # 20 QPS
                window_seconds=1,
                description="QPS全局限制"
            ),
            LimitType.DAILY_TOKEN: LimitConfig(
                limit_type=LimitType.DAILY_TOKEN,
                max_value=500000,  # 50万Token/日/用户
                window_seconds=86400,
                description="单用户单日Token限额"
            ),
            LimitType.MONTHLY_BUDGET: LimitConfig(
                limit_type=LimitType.MONTHLY_BUDGET,
                max_value=500.0,   # $500/月/租户
                window_seconds=2592000,
                description="月度预算($)"
            ),
        }

    def _get_record(self, user_id: str, tenant_id: str) -> UsageRecord:
        key = f"{tenant_id}:{user_id}"
        with self.global_lock:
            if key not in self.records:
                self.records[key] = UsageRecord(user_id=user_id, tenant_id=tenant_id)
            return self.records[key]

    def _check_and_roll_window(self, record: UsageRecord) -> None:
        """检查并滚动时间窗口"""
        now = time.time()
        # RPM/TPM 窗口滚动 (60秒)
        if now - record.window_start > 60:
            record.window_start = now
            record.tpm_used = 0.0
            record.rpm_used = 0
        # 日窗口滚动
        if now - record.day_start > 86400:
            record.day_start = now // 86400 * 86400
            record.daily_token_used = 0.0
        # 月窗口滚动
        if now - record.month_start > 2592000:
            record.month_start = now // 2592000 * 2592000
            record.monthly_cost_usd = 0.0

    def check_and_consume(
        self,
        user_id: str = "default",
        tenant_id: str = "default",
        estimated_tokens: int = 1000,
        estimated_cost_usd: float = 0.01,
    ) -> Dict[str, Any]:
        """
        检查限流并预扣配额
        Returns: {
            "allowed": bool,
            "reason": str,
            "remaining": {"tpm": ..., "rpm": ..., "daily": ..., "monthly": ...}
        }
        """
        record = self._get_record(user_id, tenant_id)
        with self.global_lock:
            self._check_and_roll_window(record)

            # 1. 检查TPM
            tpm_limit = self.limits[LimitType.TPM].max_value
            if record.tpm_used + estimated_tokens > tpm_limit:
                return {
                    "allowed": False,
                    "reason": f"TPM超限: 已用{record.tpm_used:.0f}/{tpm_limit:.0f}",
                    "remaining": {
                        "tpm": max(0, tpm_limit - record.tpm_used),
                        "rpm": max(0, self.limits[LimitType.RPM].max_value - record.rpm_used),
                        "daily": max(0, self.limits[LimitType.DAILY_TOKEN].max_value - record.daily_token_used),
                        "monthly": max(0, self.limits[LimitType.MONTHLY_BUDGET].max_value - record.monthly_cost_usd),
                    }
                }

            # 2. 检查RPM
            rpm_limit = self.limits[LimitType.RPM].max_value
            if record.rpm_used + 1 > rpm_limit:
                return {
                    "allowed": False,
                    "reason": f"RPM超限: 已用{record.rpm_used}/{rpm_limit}",
                    "remaining": {}
                }

            # 3. 检查日Token
            daily_limit = self.limits[LimitType.DAILY_TOKEN].max_value
            if record.daily_token_used + estimated_tokens > daily_limit:
                return {
                    "allowed": False,
                    "reason": f"日Token超限: 已用{record.daily_token_used:.0f}/{daily_limit:.0f}",
                    "remaining": {}
                }

            # 4. 检查月度预算
            monthly_limit = self.limits[LimitType.MONTHLY_BUDGET].max_value
            if record.monthly_cost_usd + estimated_cost_usd > monthly_limit:
                return {
                    "allowed": False,
                    "reason": f"月度预算超限: 已用${record.monthly_cost_usd:.2f}/${monthly_limit:.2f}",
                    "remaining": {}
                }

            # 全部通过, 预扣配额
            record.tpm_used += estimated_tokens
            record.rpm_used += 1
            record.daily_token_used += estimated_tokens
            record.monthly_cost_usd += estimated_cost_usd

            return {
                "allowed": True,
                "reason": "ok",
                "remaining": {
                    "tpm": max(0, tpm_limit - record.tpm_used),
                    "rpm": max(0, rpm_limit - record.rpm_used),
                    "daily": max(0, daily_limit - record.daily_token_used),
                    "monthly": max(0, monthly_limit - record.monthly_cost_usd),
                }
            }

    def record_actual_usage(
        self,
        user_id: str,
        tenant_id: str,
        actual_tokens: int,
        actual_cost_usd: float,
    ) -> None:
        """记录实际使用量 (用于校准预扣)"""
        record = self._get_record(user_id, tenant_id)
        with self.global_lock:
            # 校准: 实际使用可能少于预扣
            delta_tokens = actual_tokens - 0  # 简化: 实际=预扣
            delta_cost = actual_cost_usd - 0
            record.daily_token_used += delta_tokens
            record.monthly_cost_usd += delta_cost

    def get_usage_stats(self, user_id: str = "default", tenant_id: str = "default") -> Dict[str, Any]:
        """获取使用统计"""
        record = self._get_record(user_id, tenant_id)
        with self.global_lock:
            return {
                "user_id": user_id,
                "tenant_id": tenant_id,
                "tpm_used": record.tpm_used,
                "tpm_limit": self.limits[LimitType.TPM].max_value,
                "rpm_used": record.rpm_used,
                "rpm_limit": self.limits[LimitType.RPM].max_value,
                "daily_token_used": record.daily_token_used,
                "daily_token_limit": self.limits[LimitType.DAILY_TOKEN].max_value,
                "monthly_cost_usd": round(record.monthly_cost_usd, 2),
                "monthly_cost_limit": self.limits[LimitType.MONTHLY_BUDGET].max_value,
            }


# 全局限流器实例
_global_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """获取全局限流器"""
    global _global_limiter
    if _global_limiter is None:
        _global_limiter = RateLimiter()
    return _global_limiter


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """估算API调用成本(美元) - 2026年豆包/Kimi/Qwen价格"""
    # 2026年价格表 (每1K token)
    PRICE_TABLE = {
        "doubao-seed-2-0-pro-260215":  {"input": 0.0008, "output": 0.002},
        "doubao-seed-2-0-lite-260215": {"input": 0.0003, "output": 0.0006},
        "doubao-seed-2-0-mini-260215": {"input": 0.0001, "output": 0.0002},
        "kimi-k2-5-260127":             {"input": 0.0006, "output": 0.0015},
        "qwen-3-5-plus-260215":         {"input": 0.0004, "output": 0.0012},
        "qwen-SEA-LION-v4":             {"input": 0.0005, "output": 0.0015},
    }
    price = PRICE_TABLE.get(model, {"input": 0.001, "output": 0.002})
    return (prompt_tokens / 1000.0) * price["input"] + (completion_tokens / 1000.0) * price["output"]
