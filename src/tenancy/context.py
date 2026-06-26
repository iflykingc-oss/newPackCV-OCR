"""租户上下文 - 全链路传递租户信息

使用ContextVar实现请求级别的租户上下文隔离，
确保在高并发场景下租户信息不会串扰。
"""

import time
import uuid
from contextvars import ContextVar
from typing import Any, Dict, Optional

from tenancy.models import TenantQuota


class TenantContextMissingError(Exception):
    """租户上下文缺失异常"""
    pass


# ContextVar必须在模块级别定义，确保线程/协程隔离
_tenant_context: ContextVar[Optional[Dict[str, Any]]] = ContextVar(
    "tenant_context", default=None
)


class TenantContext:
    """租户上下文管理器

    关键设计：
    1. 使用ContextVar实现请求级别隔离（线程/协程安全）
    2. 在API入口（中间件）set，在业务节点get
    3. 包含租户ID、配额、模型权限等元信息
    """

    @staticmethod
    def set(
        tenant_id: str,
        tenant_name: str,
        tier: str,
        isolation_level: str,
        redis_namespace: str,
        quota: TenantQuota,
        allowed_models: list,
        request_id: Optional[str] = None,
    ) -> str:
        """设置当前请求的租户上下文

        Args:
            tenant_id: 租户ID
            tenant_name: 租户名称
            tier: 等级
            isolation_level: 隔离级别
            redis_namespace: Redis命名空间
            quota: 配额对象
            allowed_models: 允许的模型列表
            request_id: 请求ID（None时自动生成）

        Returns:
            request_id
        """
        if request_id is None:
            request_id = str(uuid.uuid4())

        ctx = {
            "tenant_id": tenant_id,
            "tenant_name": tenant_name,
            "tier": tier,
            "isolation_level": isolation_level,
            "redis_namespace": redis_namespace,
            "quota": quota.model_dump() if hasattr(quota, "model_dump") else quota,
            "allowed_models": allowed_models,
            "request_id": request_id,
            "timestamp": time.time(),
        }
        _tenant_context.set(ctx)
        return request_id

    @staticmethod
    def get() -> Optional[Dict[str, Any]]:
        """获取当前租户上下文"""
        return _tenant_context.get()

    @staticmethod
    def require() -> Dict[str, Any]:
        """必须获取（用于业务节点），缺失则抛异常

        Returns:
            租户上下文字典
        """
        ctx = _tenant_context.get()
        if not ctx:
            raise TenantContextMissingError(
                "租户上下文未设置。请确保请求经过AuthMiddleware处理。"
            )
        return ctx

    @staticmethod
    def get_tenant_id() -> Optional[str]:
        """快速获取租户ID"""
        ctx = _tenant_context.get()
        return ctx.get("tenant_id") if ctx else None

    @staticmethod
    def get_request_id() -> Optional[str]:
        """快速获取请求ID"""
        ctx = _tenant_context.get()
        return ctx.get("request_id") if ctx else None

    @staticmethod
    def clear() -> None:
        """清理上下文（请求结束时调用）"""
        _tenant_context.set(None)

    @staticmethod
    def is_model_allowed(model: str) -> bool:
        """检查模型是否被该租户允许使用

        Args:
            model: 模型名称

        Returns:
            True if allowed, False otherwise
        """
        ctx = _tenant_context.get()
        if not ctx:
            return False
        allowed = ctx.get("allowed_models", [])
        # 如果allowed为空列表，表示不限制
        if not allowed:
            return True
        return model in allowed
