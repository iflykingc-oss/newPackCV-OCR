"""
断路器模式（Circuit Breaker）- 熔断保护 + 自动恢复

支持 3 种状态：
  - CLOSED: 正常放行，统计失败率
  - OPEN: 熔断，直接拒绝请求
  - HALF_OPEN: 放行少量探测请求，判断是否恢复

基于滑动窗口统计失败率，到达阈值自动熔断；
冷却期后进入半开状态，探测成功则恢复。
"""
import time
import threading
from typing import Optional, Callable, Any, Dict
from enum import Enum
from collections import deque


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """
    线程安全的断路器

    Args:
        name: 断路器名称（用于日志和指标）
        failure_threshold: 触发熔断的失败数（滑动窗口内）
        success_threshold: 半开→关闭需要的连续成功数
        window_seconds: 滑动窗口时间（秒）
        cooldown_seconds: 熔断冷却时间（秒）
        half_open_max_calls: 半开状态最大探测请求数
    """

    def __init__(
        self,
        name: str = "default",
        failure_threshold: int = 5,
        success_threshold: int = 3,
        window_seconds: float = 60.0,
        cooldown_seconds: float = 30.0,
        half_open_max_calls: int = 1,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.window_seconds = window_seconds
        self.cooldown_seconds = cooldown_seconds
        self.half_open_max_calls = half_open_max_calls

        self._state = CircuitState.CLOSED
        self._last_failure_time: float = 0.0
        self._consecutive_successes: int = 0
        self._half_open_calls: int = 0

        # 滑动窗口（时间戳 + 是否失败）
        self._events: deque = deque()
        self._lock = threading.RLock()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            if self._state == CircuitState.OPEN:
                if time.time() - self._last_failure_time >= self.cooldown_seconds:
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
            return self._state

    def _cleanup_window(self) -> None:
        now = time.time()
        cutoff = now - self.window_seconds
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()

    def _failure_count(self) -> int:
        self._cleanup_window()
        return sum(1 for _, failed in self._events if failed)

    def allow_request(self) -> bool:
        """是否允许请求通过"""
        with self._lock:
            current_state = self.state
            if current_state == CircuitState.CLOSED:
                return True
            if current_state == CircuitState.OPEN:
                return False
            # HALF_OPEN
            if self._half_open_calls < self.half_open_max_calls:
                self._half_open_calls += 1
                return True
            return False

    def record_success(self) -> None:
        """记录成功"""
        with self._lock:
            self._events.append((time.time(), False))
            self._consecutive_successes += 1

            if self._state == CircuitState.HALF_OPEN:
                if self._consecutive_successes >= self.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._consecutive_successes = 0

    def record_failure(self) -> None:
        """记录失败"""
        with self._lock:
            self._events.append((time.time(), True))
            self._last_failure_time = time.time()
            self._consecutive_successes = 0

            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
            elif self._state == CircuitState.CLOSED:
                if self._failure_count() >= self.failure_threshold:
                    self._state = CircuitState.OPEN

    def get_stats(self) -> Dict[str, Any]:
        """获取断路器统计"""
        with self._lock:
            self._cleanup_window()
            total = len(self._events)
            failures = sum(1 for _, failed in self._events if failed)
            return {
                "name": self.name,
                "state": self.state.value,
                "total_requests": total,
                "failure_count": failures,
                "failure_rate": round(failures / total, 4) if total > 0 else 0.0,
                "consecutive_successes": self._consecutive_successes,
                "failure_threshold": self.failure_threshold,
                "success_threshold": self.success_threshold,
                "cooldown_seconds": self.cooldown_seconds,
                "last_failure_ago_s": round(time.time() - self._last_failure_time, 2) if self._last_failure_time else None,
            }

    def reset(self) -> None:
        """手动重置为关闭状态"""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._events.clear()
            self._consecutive_successes = 0
            self._half_open_calls = 0

    def call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """
        通过断路器调用函数
        - CLOSED/HALF_OPEN: 执行函数，记录成功/失败
        - OPEN: 抛出 CircuitOpenError
        """
        if not self.allow_request():
            raise CircuitOpenError(f"断路器 [{self.name}] 处于 OPEN 状态，请求被拒绝")

        try:
            result = func(*args, **kwargs)
            self.record_success()
            return result
        except Exception as e:
            self.record_failure()
            raise


class CircuitOpenError(Exception):
    """断路器打开时抛出的异常"""
    pass


class CircuitBreakerRegistry:
    """全局断路器注册表"""

    _instance: Optional["CircuitBreakerRegistry"] = None
    _breakers: Dict[str, CircuitBreaker] = {}

    def __new__(cls) -> "CircuitBreakerRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_or_create(
        self,
        name: str,
        failure_threshold: int = 5,
        success_threshold: int = 3,
        window_seconds: float = 60.0,
        cooldown_seconds: float = 30.0,
    ) -> CircuitBreaker:
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(
                name=name,
                failure_threshold=failure_threshold,
                success_threshold=success_threshold,
                window_seconds=window_seconds,
                cooldown_seconds=cooldown_seconds,
            )
        return self._breakers[name]

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        return {name: cb.get_stats() for name, cb in self._breakers.items()}

    def reset_all(self) -> None:
        for cb in self._breakers.values():
            cb.reset()


# 全局便捷方法
def get_circuit_registry() -> CircuitBreakerRegistry:
    return CircuitBreakerRegistry()
