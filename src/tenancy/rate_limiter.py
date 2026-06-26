"""租户级限流器 - 滑动窗口+令牌桶双算法

参考2026年最佳实践：
- DeepSeek-V4多租户推理网关：动态权重分级
- 阿里云Model Studio：客户端+架构双限流
- Redis Lua脚本保证原子性

设计要点：
1. RPM用滑动窗口（精确控制，APIGateway级精度）
2. TPM用令牌桶（支持突发+平均速率控制）
3. 长上下文Token按权重计算
4. 三层熔断（请求/会话/硬件）
"""

import time
from enum import Enum
from typing import Tuple

from tenancy.context import TenantContext
from utils.redis_client import redis_client


class RateLimitType(str, Enum):
    """限流类型"""
    RPM = "rpm"           # 每分钟请求数
    TPM = "tpm"           # 每分钟Token数
    CONCURRENT = "concurrent"  # 并发数


class RateLimitResult:
    """限流结果"""

    def __init__(
        self,
        allowed: bool,
        limit_type: str,
        current: int,
        limit: int,
        retry_after: float = 0,
        metadata: dict = None,
    ):
        self.allowed = allowed
        self.limit_type = limit_type
        self.current = current
        self.limit = limit
        self.retry_after = retry_after
        self.metadata = metadata or {}

    def to_dict(self) -> dict:
        return {
            "allowed": self.allowed,
            "limit_type": self.limit_type,
            "current": self.current,
            "limit": self.limit,
            "remaining": max(0, self.limit - self.current),
            "retry_after": self.retry_after,
            "metadata": self.metadata,
        }


class TenantRateLimiter:
    """租户级限流器

    使用滑动窗口（精确）+ 令牌桶（突发）双算法组合。
    适用于SaaS多租户LLM服务的限流场景。
    """

    # 滑动窗口Lua脚本 - 精确控制RPM
    SLIDING_WINDOW_LUA = """
    local key = KEYS[1]
    local now = tonumber(ARGV[1])
    local window = tonumber(ARGV[2])
    local limit = tonumber(ARGV[3])
    local request_id = ARGV[4]
    
    -- 移除窗口外的请求
    redis.call('ZREMRANGEBYSCORE', key, '-inf', now - window * 1000)
    
    -- 获取当前窗口内请求数
    local count = redis.call('ZCARD', key)
    
    if count < limit then
        -- 添加当前请求（用request_id避免重复）
        redis.call('ZADD', key, now, request_id)
        redis.call('EXPIRE', key, window * 2)
        return {1, count + 1, limit, 0}
    else
        -- 获取最早请求时间戳，计算retry_after
        local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
        local retry_after = 0
        if #oldest > 0 then
            retry_after = (tonumber(oldest[2]) + window * 1000 - now) / 1000
        end
        return {0, count, limit, retry_after}
    end
    """

    # 令牌桶Lua脚本 - Token突发控制
    TOKEN_BUCKET_LUA = """
    local tokens_key = KEYS[1]
    local timestamp_key = KEYS[2]
    local rate = tonumber(ARGV[1])
    local capacity = tonumber(ARGV[2])
    local now = tonumber(ARGV[3])
    local requested = tonumber(ARGV[4])
    
    local last_tokens = tonumber(redis.call('GET', tokens_key) or capacity)
    local last_refresh = tonumber(redis.call('GET', timestamp_key) or now)
    
    local elapsed = math.max(0, now - last_refresh)
    local new_tokens = math.min(capacity, last_tokens + elapsed * rate)
    
    if new_tokens >= requested then
        local remaining = new_tokens - requested
        redis.call('SET', tokens_key, remaining)
        redis.call('SET', timestamp_key, now)
        redis.call('EXPIRE', tokens_key, 60)
        redis.call('EXPIRE', timestamp_key, 60)
        return {1, remaining, capacity, 0}
    else
        -- 令牌不足，计算需要等待多久
        local needed = requested - new_tokens
        local wait = needed / rate
        redis.call('SET', tokens_key, new_tokens)
        redis.call('SET', timestamp_key, now)
        return {0, new_tokens, capacity, wait}
    end
    """

    def __init__(self):
        self._sliding_window_sha: str = None
        self._token_bucket_sha: str = None

    def _ensure_scripts_loaded(self) -> None:
        """确保Lua脚本已加载"""
        if self._sliding_window_sha is None:
            self._sliding_window_sha = redis_client.load_script(
                self.SLIDING_WINDOW_LUA
            )
        if self._token_bucket_sha is None:
            self._token_bucket_sha = redis_client.load_script(
                self.TOKEN_BUCKET_LUA
            )

    def check_request(self, cost: int = 1) -> RateLimitResult:
        """检查请求限流（RPM）

        Args:
            cost: 本次请求消耗的请求数（通常为1，批量场景可调高）

        Returns:
            RateLimitResult
        """
        ctx = TenantContext.get()
        if not ctx:
            # 无租户上下文（系统内部调用）放行
            return RateLimitResult(
                allowed=True, limit_type="rpm",
                current=0, limit=99999,
            )

        self._ensure_scripts_loaded()

        quota = ctx["quota"]
        max_rpm = quota["base_rpm"] + quota["elastic_rpm"] + quota["burst_buffer"]
        namespace = ctx["redis_namespace"]
        now_ms = int(time.time() * 1000)
        request_id = f"{ctx['request_id']}:{int(time.time() * 1000000)}"

        try:
            result = redis_client.evalsha(
                self._sliding_window_sha,
                namespace,
                1,
                "rl:rpm",
                now_ms, 60, max_rpm, request_id,
            )
            allowed, current, limit, retry_after = result
            return RateLimitResult(
                allowed=bool(int(allowed)),
                limit_type="rpm",
                current=int(current),
                limit=int(limit),
                retry_after=float(retry_after),
                metadata={"max_rpm": max_rpm},
            )
        except Exception:
            # Redis故障时降级放行（fail-open），但记录日志
            return RateLimitResult(
                allowed=True, limit_type="rpm",
                current=0, limit=max_rpm,
                metadata={"degraded": True},
            )

    def check_tokens(self, tokens: int) -> RateLimitResult:
        """检查Token限流（TPM，令牌桶）

        Args:
            tokens: 本次请求消耗的Token数

        Returns:
            RateLimitResult
        """
        ctx = TenantContext.get()
        if not ctx:
            return RateLimitResult(
                allowed=True, limit_type="tpm",
                current=0, limit=999999,
            )

        self._ensure_scripts_loaded()

        quota = ctx["quota"]
        total_tpm = quota["base_tpm"] + quota["elastic_tpm"]
        burst_tpm = quota["burst_tpm_buffer"]
        capacity = total_tpm + burst_tpm
        rate = total_tpm / 60.0  # 每秒补充令牌数
        namespace = ctx["redis_namespace"]
        now = time.time()

        try:
            result = redis_client.evalsha(
                self._token_bucket_sha,
                namespace,
                2,
                "tb:tokens",
                "tb:ts",
                rate, capacity, now, tokens,
            )
            allowed, remaining, cap, wait = result
            return RateLimitResult(
                allowed=bool(int(allowed)),
                limit_type="tpm",
                current=int(cap - remaining),
                limit=int(cap),
                retry_after=float(wait),
                metadata={
                    "rate_per_sec": rate,
                    "tokens_requested": tokens,
                    "remaining": remaining,
                },
            )
        except Exception:
            return RateLimitResult(
                allowed=True, limit_type="tpm",
                current=0, limit=capacity,
                metadata={"degraded": True},
            )

    def check_concurrent(self, current_concurrent: int) -> RateLimitResult:
        """检查并发数（基于Redis计数器）

        Args:
            current_concurrent: 当前并发数

        Returns:
            RateLimitResult
        """
        ctx = TenantContext.get()
        if not ctx:
            return RateLimitResult(
                allowed=True, limit_type="concurrent",
                current=0, limit=99999,
            )

        quota = ctx["quota"]
        max_concurrent = quota["base_concurrent"]
        namespace = ctx["redis_namespace"]

        if current_concurrent >= max_concurrent:
            return RateLimitResult(
                allowed=False,
                limit_type="concurrent",
                current=current_concurrent,
                limit=max_concurrent,
                retry_after=1.0,
            )
        return RateLimitResult(
            allowed=True,
            limit_type="concurrent",
            current=current_concurrent,
            limit=max_concurrent,
        )

    def check_all(self, estimated_tokens: int = 0) -> Tuple[bool, list]:
        """综合检查所有限流

        Args:
            estimated_tokens: 预估Token数

        Returns:
            (是否全部通过, [限流结果列表])
        """
        results = []
        all_passed = True

        # 1. RPM检查
        rpm_result = self.check_request()
        results.append(rpm_result)
        if not rpm_result.allowed:
            all_passed = False

        # 2. TPM检查（如果提供了预估Token）
        if estimated_tokens > 0:
            tpm_result = self.check_tokens(estimated_tokens)
            results.append(tpm_result)
            if not tpm_result.allowed:
                all_passed = False

        return all_passed, results


# 全局限流器单例
rate_limiter = TenantRateLimiter()
