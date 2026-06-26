"""限流降级策略模块"""
import time
import logging
from typing import Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class DegradationLevel(str, Enum):
    """降级等级"""
    NORMAL = "normal"               # 正常：使用高质量模型
    LITE = "lite"                   # 降级1：使用轻量模型
    MINI = "mini"                   # 降级2：使用mini模型
    CACHED = "cached"               # 降级3：返回缓存
    REJECT = "reject"               # 拒绝：直接返回429


@dataclass
class DegradationDecision:
    """降级决策"""
    level: DegradationLevel
    target_model: Optional[str] = None
    reason: str = ""
    retry_after: int = 0
    use_cache: bool = False
    cache_key: Optional[str] = None


class FallbackPolicy:
    """限流降级策略引擎

    参考DeepSeek-V4设计：
    - 3层熔断（请求级/会话级/硬件级）
    - 5级Fallback链
    - 智能降级（从高成本到低成本）
    """

    # 模型降级链（高质量 → 低质量）
    MODEL_FALLBACK_CHAIN = {
        "doubao-seed-2-0-pro-260215": [
            "kimi-k2-5-260127",
            "qwen-3-5-plus-260215",
            "doubao-seed-2-0-lite-260215",
            "doubao-seed-2-0-mini-260215",
        ],
        "kimi-k2-5-260127": [
            "qwen-3-5-plus-260215",
            "doubao-seed-2-0-lite-260215",
            "doubao-seed-2-0-mini-260215",
        ],
        "qwen-3-5-plus-260215": [
            "doubao-seed-2-0-lite-260215",
            "doubao-seed-2-0-mini-260215",
        ],
        "doubao-seed-2-0-lite-260215": [
            "doubao-seed-2-0-mini-260215",
        ],
    }

    # 降级阈值
    THRESHOLDS = {
        "tpm_usage_warning": 0.7,         # TPM使用70%告警
        "tpm_usage_degrade_lite": 0.85,    # 85%降级到lite
        "tpm_usage_degrade_mini": 0.95,    # 95%降级到mini
        "tpm_usage_reject": 1.0,           # 100%拒绝
        "error_rate_degrade": 0.1,         # 错误率10%触发降级
    }

    @classmethod
    def decide(
        cls,
        current_model: str,
        tpm_used: int,
        tpm_limit: int,
        error_rate: float = 0.0,
        session_count: int = 0,
    ) -> DegradationDecision:
        """根据当前状态决定降级策略

        Args:
            current_model: 当前模型
            tpm_used: 已用Token
            tpm_limit: TPM上限
            error_rate: 错误率（0-1）
            session_count: 当前会话数

        Returns:
            降级决策
        """
        usage_ratio = tpm_used / tpm_limit if tpm_limit > 0 else 0

        # 1. 检查错误率（最优先）
        if error_rate >= cls.THRESHOLDS["error_rate_degrade"]:
            # 错误率过高，降级模型
            chain = cls.MODEL_FALLBACK_CHAIN.get(current_model, [])
            if chain:
                return DegradationDecision(
                    level=DegradationLevel.LITE,
                    target_model=chain[0],
                    reason=f"错误率{error_rate:.1%}过高，降级到{chain[0]}",
                    retry_after=0
                )

        # 2. 检查TPM使用率
        if usage_ratio >= cls.THRESHOLDS["tpm_usage_reject"]:
            return DegradationDecision(
                level=DegradationLevel.REJECT,
                reason=f"TPM使用率{usage_ratio:.1%}，已满",
                retry_after=60
            )

        if usage_ratio >= cls.THRESHOLDS["tpm_usage_degrade_mini"]:
            chain = cls.MODEL_FALLBACK_CHAIN.get(current_model, [])
            if len(chain) >= 2:
                return DegradationDecision(
                    level=DegradationLevel.MINI,
                    target_model=chain[-1],
                    reason=f"TPM使用率{usage_ratio:.1%}，降级到mini",
                    retry_after=0
                )

        if usage_ratio >= cls.THRESHOLDS["tpm_usage_degrade_lite"]:
            chain = cls.MODEL_FALLBACK_CHAIN.get(current_model, [])
            if chain:
                return DegradationDecision(
                    level=DegradationLevel.LITE,
                    target_model=chain[0],
                    reason=f"TPM使用率{usage_ratio:.1%}，降级到lite",
                    retry_after=0
                )

        # 3. 检查会话数
        if session_count > 1000:
            return DegradationDecision(
                level=DegradationLevel.CACHED,
                reason=f"会话数{session_count}过高，启用缓存",
                use_cache=True,
                cache_key=f"session:{session_count}:{int(time.time() // 60)}"
            )

        # 4. 正常
        return DegradationDecision(
            level=DegradationLevel.NORMAL,
            target_model=current_model,
            reason="正常状态"
        )

    @classmethod
    def should_warn(cls, tpm_used: int, tpm_limit: int) -> bool:
        """是否需要预警"""
        ratio = tpm_used / tpm_limit if tpm_limit > 0 else 0
        return ratio >= cls.THRESHOLDS["tpm_usage_warning"]

    @classmethod
    def get_fallback_chain(cls, model: str) -> list:
        """获取模型的降级链"""
        return cls.MODEL_FALLBACK_CHAIN.get(model, [])
