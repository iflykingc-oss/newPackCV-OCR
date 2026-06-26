"""实时仪表盘 + 限流可视化路由"""
from fastapi import APIRouter, Query
from typing import Optional

from monitoring.realtime_dashboard import dashboard, RateLimitSnapshot

router = APIRouter(prefix="/monitoring", tags=["实时仪表盘"])


@router.get("/overview")
async def dashboard_overview():
    """仪表盘概览"""
    return dashboard.get_overview()


@router.get("/latency")
async def latency_percentiles(endpoint: Optional[str] = Query(None)):
    """延迟分位统计"""
    return dashboard.get_latency_percentiles(endpoint)


@router.get("/top-endpoints")
async def top_endpoints(n: int = Query(10, ge=1, le=50)):
    """TopN 热点端点"""
    return {"top": dashboard.get_top_endpoints(n)}


@router.get("/rate-limits")
async def rate_limit_status():
    """租户限流状态"""
    return {"tenants": dashboard.get_rate_limit_status()}


@router.get("/heatmap")
async def heatmap(last_minutes: int = Query(60, ge=1, le=1440)):
    """请求热力图数据"""
    return {"data": dashboard.get_heatmap_data(last_minutes)}


@router.post("/record")
async def record_sample(endpoint: str, duration_ms: float, status_code: int = 200):
    """手动记录延迟采样（测试用）"""
    dashboard.record_latency(endpoint, duration_ms, status_code)
    return {"recorded": True, "endpoint": endpoint, "duration_ms": duration_ms}


@router.post("/rate-limit-snapshot")
async def update_rate_limit(tenant_id: str, rpm_current: int = 0, rpm_limit: int = 300,
                            tpm_current: int = 0, tpm_limit: int = 500000,
                            concurrent_current: int = 0, concurrent_limit: int = 20,
                            blocked_count: int = 0):
    """更新租户限流快照"""
    snap = RateLimitSnapshot(
        tenant_id=tenant_id,
        rpm_current=rpm_current,
        rpm_limit=rpm_limit,
        tpm_current=tpm_current,
        tpm_limit=tpm_limit,
        concurrent_current=concurrent_current,
        concurrent_limit=concurrent_limit,
        blocked_count=blocked_count,
    )
    dashboard.update_rate_limit(snap)
    return {"updated": True, "tenant_id": tenant_id}


@router.post("/reset")
async def reset_dashboard():
    """重置仪表盘数据"""
    dashboard.reset()
    return {"reset": True}
