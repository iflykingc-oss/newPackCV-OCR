"""灰度发布 / 金丝雀部署"""
import hashlib
import time
import uuid
from enum import Enum
from typing import Dict, List, Optional, Set
from pydantic import BaseModel, Field


class RolloutStrategy(str, Enum):
    """发布策略"""
    PERCENTAGE = "percentage"   # 百分比
    WHITELIST = "whitelist"     # 白名单
    HEADER = "header"           # Header 强制
    REGION = "region"           # 地域


class CanaryConfig(BaseModel):
    """金丝雀配置"""
    canary_id: str = Field(default_factory=lambda: f"canary-{uuid.uuid4().hex[:8]}")
    name: str
    version: str
    strategy: RolloutStrategy = RolloutStrategy.PERCENTAGE
    percentage: int = Field(default=0, ge=0, le=100)
    whitelist: Set[str] = Field(default_factory=set)
    header_value: Optional[str] = None
    regions: Set[str] = Field(default_factory=set)
    enabled: bool = True
    created_at: float = Field(default_factory=time.time)


class CanaryDeployer:
    """灰度发布器"""

    def __init__(self):
        self._canaries: Dict[str, CanaryConfig] = {}

    def create_canary(
        self,
        name: str,
        version: str,
        strategy: RolloutStrategy = RolloutStrategy.PERCENTAGE,
        percentage: int = 0,
        whitelist: Optional[Set[str]] = None,
        header_value: Optional[str] = None,
        regions: Optional[Set[str]] = None,
    ) -> CanaryConfig:
        """创建金丝雀"""
        config = CanaryConfig(
            name=name,
            version=version,
            strategy=strategy,
            percentage=percentage,
            whitelist=whitelist or set(),
            header_value=header_value,
            regions=regions or set(),
        )
        self._canaries[config.canary_id] = config
        return config

    def route(
        self,
        user_id: str,
        headers: Optional[Dict[str, str]] = None,
        region: Optional[str] = None,
    ) -> Optional[str]:
        """根据规则路由到版本"""
        headers = headers or {}

        for canary in self._canaries.values():
            if not canary.enabled:
                continue
            if canary.strategy == RolloutStrategy.WHITELIST:
                if user_id in canary.whitelist:
                    return canary.version
            elif canary.strategy == RolloutStrategy.HEADER:
                if headers.get("X-Canary") == canary.header_value:
                    return canary.version
            elif canary.strategy == RolloutStrategy.REGION:
                if region and region in canary.regions:
                    return canary.version
            elif canary.strategy == RolloutStrategy.PERCENTAGE:
                h = int(hashlib.md5(
                    f"{canary.canary_id}:{user_id}".encode()
                ).hexdigest(), 16) % 100
                if h < canary.percentage:
                    return canary.version
        return None

    def promote(self, canary_id: str, new_percentage: int) -> bool:
        """提升灰度比例"""
        c = self._canaries.get(canary_id)
        if not c:
            return False
        c.percentage = max(0, min(100, new_percentage))
        return True

    def rollback(self, canary_id: str) -> bool:
        """回滚"""
        c = self._canaries.get(canary_id)
        if not c:
            return False
        c.enabled = False
        c.percentage = 0
        return True

    def list_canaries(self) -> List[CanaryConfig]:
        return list(self._canaries.values())
