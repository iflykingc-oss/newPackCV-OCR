"""重试策略（指数退避）

支持可配置的最大重试次数、基础延迟和上限。
"""
import random
from dataclasses import dataclass
from typing import List


@dataclass
class RetryPolicy:
    """重试策略

    默认策略：1s, 5s, 30s, 2min, 10min（最多 5 次重试，共 6 次尝试）
    """
    max_attempts: int = 6
    base_delay: float = 1.0  # 基础延迟（秒）
    max_delay: float = 600.0  # 最大延迟（秒，10 分钟）
    multiplier: float = 5.0  # 退避倍数
    jitter: float = 0.1  # 抖动比例（±10%）

    def get_delays(self) -> List[float]:
        """获取所有重试延迟序列（不含首次投递）

        Returns:
            延迟列表（秒），长度 = max_attempts - 1
        """
        delays: List[float] = []
        delay = self.base_delay
        for _ in range(self.max_attempts - 1):
            jitter_amount = delay * self.jitter
            actual_delay = delay + random.uniform(-jitter_amount, jitter_amount)
            delays.append(max(0.0, actual_delay))
            delay = min(delay * self.multiplier, self.max_delay)
        return delays


class ExponentialBackoff:
    """指数退避计算器"""

    def __init__(self, base: float = 1.0, max_delay: float = 600.0, multiplier: float = 5.0):
        self.base = base
        self.max_delay = max_delay
        self.multiplier = multiplier

    def compute(self, attempt: int) -> float:
        """计算第 N 次重试的延迟

        Args:
            attempt: 重试次数（1 表示第一次重试）

        Returns:
            延迟秒数
        """
        delay = self.base * (self.multiplier ** (attempt - 1))
        return min(delay, self.max_delay)
