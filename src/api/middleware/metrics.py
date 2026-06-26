"""
PackCV-OCR 监控集成中间件
将API metrics、计费、限流、降级、审计等业务事件接入Prometheus
"""
import time
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from api.metrics import (
    API_REQUESTS_TOTAL,
    API_REQUEST_LATENCY,
    API_REQUEST_SIZE,
    API_RESPONSE_SIZE,
    RATE_LIMIT_HITS_TOTAL,
    record_api_request,
    record_rate_limit_hit,
    record_workflow_execution,
    record_fallback,
    record_billing_tokens,
    record_data_masking,
    record_audit,
)
from tenancy.context import TenantContext


class MetricsMiddleware(BaseHTTPMiddleware):
    """API metrics埋点中间件"""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()

        # 请求体大小
        request_size = 0
        if request.headers.get("content-length"):
            try:
                request_size = int(request.headers.get("content-length"))
            except (ValueError, TypeError):
                pass

        # 执行业务
        try:
            response = await call_next(request)
        except Exception as e:
            # 异常也记录
            elapsed = time.time() - start_time
            ctx = TenantContext.get() or {}
            tenant_id = ctx.get("tenant_id", "anonymous")
            endpoint = self._get_endpoint_pattern(request)
            method = request.method
            if API_REQUESTS_TOTAL:
                API_REQUESTS_TOTAL.labels(
                    tenant_id=tenant_id, endpoint=endpoint, method=method, status="500"
                ).inc()
            raise

        # 延迟计算
        elapsed = time.time() - start_time

        # 获取租户信息
        ctx = TenantContext.get() or {}
        tenant_id = ctx.get("tenant_id", "anonymous")
        endpoint = self._get_endpoint_pattern(request)
        method = request.method
        status = response.status_code

        # 记录API指标
        record_api_request(tenant_id, endpoint, method, status)

        if API_REQUEST_LATENCY:
            API_REQUEST_LATENCY.labels(endpoint=endpoint, method=method).observe(elapsed)

        if API_REQUEST_SIZE and request_size > 0:
            API_REQUEST_SIZE.labels(endpoint=endpoint).observe(request_size)

        if API_RESPONSE_SIZE:
            response_size = 0
            if hasattr(response, "headers"):
                cl = response.headers.get("content-length")
                if cl:
                    try:
                        response_size = int(cl)
                    except (ValueError, TypeError):
                        pass
            if response_size > 0:
                API_RESPONSE_SIZE.labels(endpoint=endpoint).observe(response_size)

        # 限流命中
        if status == 429:
            record_rate_limit_hit(tenant_id, "request")

        return response

    @staticmethod
    def _get_endpoint_pattern(request: Request) -> str:
        """获取路由模式（避免高基数）"""
        route = request.scope.get("route")
        if route and hasattr(route, "path"):
            return route.path
        return request.url.path


# ============================================================
# FastAPI 路由
# ============================================================

async def metrics_endpoint(request: Request) -> Response:
    """Prometheus metrics端点"""
    from api.metrics import get_metrics
    return Response(
        content=get_metrics(),
        media_type="text/plain; version=0.0.4; charset=utf-8"
    )
