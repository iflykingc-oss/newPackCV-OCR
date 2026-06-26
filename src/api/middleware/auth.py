"""鉴权中间件 - 提取租户上下文 + 限流

参考DeepSeek-V4最佳实践：
- X-Tenant-Info 头传递租户ID和配额标识（避免下游重复验签）
- max-age=0 避免API网关5-10分钟密钥缓存延迟
- 三级限流：RPM + TPM + 并发
"""

import json
import time
import uuid
from typing import Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from tenancy.api_key_manager import APIKeyManager
from tenancy.context import TenantContext
from tenancy.rate_limiter import rate_limiter


# 不需要鉴权的路径（精确匹配）
PUBLIC_PATHS = {
    "/",
    "/health",
    "/api/v1/health",
    "/docs",
    "/openapi.json",
    "/redoc",
    "/metrics",  # Prometheus metrics
    "/api/v1/admin/tenants",  # 租户管理（管理员）
    "/api/v1/admin/tenants/demo",  # 创建演示租户
    # Web 管理后台页面（精确匹配，不能放前缀否则 / 匹配所有）
    "/tenants", "/usage", "/billing", "/settings", "/openapi-spec",
}

# 公共路径前缀（注意：匹配 APIRouter 的实际路径，不含 /api/v1）
PUBLIC_PREFIXES = (
    "/docs", "/openapi", "/redoc", "/static",
    "/api/v1/admin/tenants",
    "/api/v1/health", "/api/v1/system",
    # Web 管理后台（5个页面 + 静态资源）— 精确匹配而非前缀
    # 注意："/" 前缀会匹配所有路径，已在 PUBLIC_PATHS 中处理
    # Phase 3 新增公开端点
    "/admin/dashboard",   # 管理员 Dashboard
    "/providers",         # LLM Provider 查询
    "/webhooks",          # WebHook 管理
    "/openapi.json",      # OpenAPI 规范
    "/health",            # K8s 三探针（live/ready/startup）
    # Phase 5 新增公开端点
    "/i18n",              # 国际化
    "/intelligence",      # LLM 缓存 + Few-shot
    "/rbac",              # RBAC
    "/sso",               # OIDC SSO
    "/canary",            # 灰度发布
    "/streaming",         # SSE 流式
    # Phase 6 新增公开端点
    "/graphql",           # GraphQL API
    "/api-lifecycle",     # API 版本管理
    "/monitoring",        # 实时仪表盘
    "/admin/web",         # Web 后台（与 / 区分开）
    # Phase 7 新增公开端点（管理功能，需要内部保护但此处为演示开放）
    "/docs-enhanced",     # API 文档增强
    "/errors",            # 错误码管理
    "/config",            # 配置热更新
    "/lineage",           # 数据血缘
    "/tracing",           # 分布式追踪
)


class AuthMiddleware(BaseHTTPMiddleware):
    """鉴权+限流中间件

    流程：
    1. 检查路径是否在白名单
    2. 提取 X-API-Key 和 X-API-Secret
    3. 验证租户身份
    4. 设置租户上下文
    5. 执行限流检查
    6. 业务处理
    7. 设置响应头（追踪）
    """

    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        # 1. 健康检查和文档跳过
        path = request.url.path.rstrip("/")
        if (
            path in PUBLIC_PATHS
            or path == ""
            or any(path.startswith(p) for p in PUBLIC_PREFIXES)
        ):
            return await call_next(request)

        # 2. 提取凭证
        api_key = request.headers.get("X-API-Key")
        api_secret = request.headers.get("X-API-Secret")

        if not api_key:
            return JSONResponse(
                status_code=401,
                content={
                    "error": "missing_api_key",
                    "message": "请提供 X-API-Key 请求头",
                },
            )

        # 3. 验证租户
        tenant = None
        if api_secret:
            # 完整验证（API Key + Secret）
            tenant = APIKeyManager.verify_api_key(api_key, api_secret)
        else:
            # 仅API Key（适用于SDK自动签名场景）
            tenant = APIKeyManager.verify_api_key_only(api_key)

        if tenant is None:
            return JSONResponse(
                status_code=401,
                content={
                    "error": "invalid_credentials",
                    "message": "API Key或Secret无效",
                },
            )

        # 4. 提取或生成Request ID
        request_id = request.headers.get("X-Request-Id", str(uuid.uuid4()))

        # 5. 设置租户上下文
        TenantContext.set(
            tenant_id=tenant.tenant_id,
            tenant_name=tenant.tenant_name,
            tier=tenant.tier if isinstance(tenant.tier, str) else tenant.tier.value,
            isolation_level=(
                tenant.isolation_level
                if isinstance(tenant.isolation_level, str)
                else tenant.isolation_level.value
            ),
            redis_namespace=tenant.redis_namespace,
            quota=tenant.quota,
            allowed_models=tenant.allowed_models,
            request_id=request_id,
        )

        try:
            # 6. 限流检查（RPM）
            rate_result = rate_limiter.check_request()
            if not rate_result.allowed:
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "rate_limit_exceeded",
                        "message": f"RPM限流：当前{rate_result.current}/{rate_result.limit}",
                        "retry_after": rate_result.retry_after,
                    },
                    headers={
                        "Retry-After": str(int(rate_result.retry_after) + 1),
                        "X-RateLimit-Limit": str(rate_result.limit),
                        "X-RateLimit-Remaining": "0",
                    },
                )

            # 7. 执行业务
            response = await call_next(request)

            # 8. 设置响应头（追踪+限流信息）
            response.headers["X-Request-Id"] = request_id
            response.headers["X-Tenant-Id"] = tenant.tenant_id
            response.headers["X-RateLimit-Limit"] = str(rate_result.limit)
            response.headers["X-RateLimit-Remaining"] = str(
                max(0, rate_result.limit - rate_result.current)
            )
            response.headers["X-RateLimit-Reset"] = str(int(time.time()) + 60)

            return response
        finally:
            # 9. 清理租户上下文
            TenantContext.clear()
