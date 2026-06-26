"""API 版本管理 + 废弃策略

功能：
- 多版本路由注册（/v1/, /v2/ 共存）
- 端点废弃追踪（sunset date + alternate）
- 版本协商（Header: Accept-Version / X-API-Version）
- 废弃告警（响应头 Sunset / Deprecation / Link）
"""
import time
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from enum import Enum


class DeprecationLevel(Enum):
    """废弃级别"""
    ACTIVE = "active"           # 正常使用
    DEPRECATED = "deprecated"   # 已废弃但仍可用
    SUNSET = "sunset"          # 即将下线（返回警告）
    REMOVED = "removed"        # 已移除（返回 410 Gone）


@dataclass
class EndpointInfo:
    """端点版本信息"""
    path: str
    method: str
    version: str = "v1"
    deprecated_since: Optional[str] = None     # 废弃日期 ISO8601
    sunset_date: Optional[str] = None          # 下线日期 ISO8601
    alternate_path: Optional[str] = None       # 替代端点
    deprecation_level: DeprecationLevel = DeprecationLevel.ACTIVE
    description: str = ""
    migration_guide: str = ""


class APIVersionManager:
    """API 版本管理器
    
    用法：
        manager = APIVersionManager()
        manager.register("/api/v1/extract", "POST", version="v1")
        manager.deprecate("/api/v1/extract", "POST", 
                          alternate="/api/v2/extract",
                          sunset_date="2026-09-01")
    """

    def __init__(self) -> None:
        self._endpoints: Dict[str, EndpointInfo] = {}
        self._version_aliases: Dict[str, str] = {}  # v1 -> v1, stable -> v2
        self._current_version: str = "v1"

    def register(
        self,
        path: str,
        method: str,
        version: str = "v1",
        description: str = "",
    ) -> EndpointInfo:
        """注册端点"""
        key = f"{method.upper()}:{path}"
        info = EndpointInfo(
            path=path, method=method.upper(),
            version=version, description=description,
        )
        self._endpoints[key] = info
        return info

    def deprecate(
        self,
        path: str,
        method: str,
        alternate_path: Optional[str] = None,
        deprecated_since: Optional[str] = None,
        sunset_date: Optional[str] = None,
        migration_guide: str = "",
    ) -> Optional[EndpointInfo]:
        """标记端点为废弃"""
        key = f"{method.upper()}:{path}"
        info = self._endpoints.get(key)
        if info is None:
            return None
        info.deprecated_since = deprecated_since or _iso_now()
        info.sunset_date = sunset_date
        info.alternate_path = alternate_path
        info.migration_guide = migration_guide
        
        # 如果 sunset date 已过，标记为 REMOVED
        if sunset_date and _iso_now() > sunset_date:
            info.deprecation_level = DeprecationLevel.REMOVED
        elif sunset_date:
            info.deprecation_level = DeprecationLevel.SUNSET
        else:
            info.deprecation_level = DeprecationLevel.DEPRECATED
        return info

    def check(self, path: str, method: str) -> Optional[EndpointInfo]:
        """检查端点状态"""
        key = f"{method.upper()}:{path}"
        return self._endpoints.get(key)

    def set_version_alias(self, alias: str, target: str) -> None:
        """设置版本别名（如 stable -> v2）"""
        self._version_aliases[alias] = target

    def resolve_version(self, version: str) -> str:
        """解析版本别名"""
        return self._version_aliases.get(version, version)

    def set_current_version(self, version: str) -> None:
        """设置当前默认版本"""
        self._current_version = version

    def get_current_version(self) -> str:
        return self._current_version

    def list_endpoints(
        self,
        version: Optional[str] = None,
        level: Optional[DeprecationLevel] = None,
    ) -> List[EndpointInfo]:
        """列出端点，可按版本/废弃级别过滤"""
        result = list(self._endpoints.values())
        if version:
            result = [e for e in result if e.version == version]
        if level:
            result = [e for e in result if e.deprecation_level == level]
        return result

    def get_deprecation_headers(self, path: str, method: str) -> Dict[str, str]:
        """获取废弃相关的响应头"""
        info = self.check(path, method)
        if info is None or info.deprecation_level == DeprecationLevel.ACTIVE:
            return {}
        headers: Dict[str, str] = {}
        if info.deprecation_level in (DeprecationLevel.DEPRECATED, DeprecationLevel.SUNSET):
            headers["Deprecation"] = "true"
        if info.sunset_date:
            headers["Sunset"] = info.sunset_date
        if info.alternate_path:
            headers["Link"] = f'<{info.alternate_path}>; rel="successor-version"'
        if info.deprecated_since:
            headers["X-Deprecated-Since"] = info.deprecated_since
        if info.migration_guide:
            headers["X-Migration-Guide"] = info.migration_guide
        return headers

    def stats(self) -> Dict[str, int]:
        """统计"""
        counts: Dict[str, int] = {}
        for info in self._endpoints.values():
            level = info.deprecation_level.value
            counts[level] = counts.get(level, 0) + 1
        return counts

    def bulk_register(self, endpoints: List[Dict[str, str]]) -> int:
        """批量注册端点"""
        count = 0
        for ep in endpoints:
            self.register(
                path=ep.get("path", ""),
                method=ep.get("method", "GET"),
                version=ep.get("version", "v1"),
                description=ep.get("description", ""),
            )
            count += 1
        return count


# 单例
version_manager = APIVersionManager()


def _iso_now() -> str:
    """返回当前 ISO 日期"""
    import datetime
    return datetime.datetime.utcnow().strftime("%Y-%m-%d")


def init_default_endpoints() -> None:
    """初始化默认端点注册"""
    version_manager.bulk_register([
        {"path": "/api/v1/extract", "method": "POST", "version": "v1", "description": "信息提取"},
        {"path": "/api/v1/qa", "method": "POST", "version": "v1", "description": "QA问答"},
        {"path": "/api/v1/batch", "method": "POST", "version": "v1", "description": "批量提取"},
        {"path": "/api/v1/scenarios", "method": "GET", "version": "v1", "description": "场景列表"},
        {"path": "/api/v1/health", "method": "GET", "version": "v1", "description": "健康检查"},
        {"path": "/api/v1/me", "method": "GET", "version": "v1", "description": "租户信息"},
        {"path": "/api/v1/billing/usage", "method": "GET", "version": "v1", "description": "用量查询"},
        {"path": "/api/v1/billing/invoice", "method": "GET", "version": "v1", "description": "账单查询"},
        {"path": "/api/v1/billing/record", "method": "POST", "version": "v1", "description": "计费记录"},
        {"path": "/api/v1/audit/logs", "method": "GET", "version": "v1", "description": "审计日志"},
        {"path": "/api/v1/security/mask", "method": "POST", "version": "v1", "description": "数据脱敏"},
        {"path": "/api/v1/security/validate", "method": "POST", "version": "v1", "description": "安全校验"},
        {"path": "/api/v1/degradation/policy", "method": "GET", "version": "v1", "description": "降级策略"},
        {"path": "/api/v1/usage", "method": "GET", "version": "v1", "description": "配额用量"},
    ])
    # 标记即将废弃的端点
    version_manager.deprecate(
        "/api/v1/usage", "GET",
        alternate_path="/api/v1/billing/usage",
        sunset_date="2026-12-01",
        migration_guide="使用 /api/v1/billing/usage 替代，返回更详细的用量数据",
    )
    # 版本别名
    version_manager.set_version_alias("stable", "v1")
    version_manager.set_version_alias("latest", "v1")


from typing import Dict
