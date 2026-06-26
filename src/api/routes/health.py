"""
健康探针路由 - K8s 三探针 + 断路器状态
"""
from fastapi import APIRouter
from resilience import get_health_probe, get_circuit_registry

router = APIRouter(prefix="/health", tags=["Health Probes"])


@router.get("/live")
async def liveness():
    """Liveness 探针：进程是否存活"""
    probe = get_health_probe()
    result = probe.liveness()
    status_code = 200 if result["status"] == "ok" else 503
    return result


@router.get("/ready")
async def readiness():
    """Readiness 探针：是否可接收流量"""
    probe = get_health_probe()
    result = probe.readiness()
    status_code = 200 if result["status"] == "ok" else 503
    return result


@router.get("/startup")
async def startup():
    """Startup 探针：启动是否完成"""
    probe = get_health_probe()
    result = probe.startup()
    status_code = 200 if result["status"] == "ok" else 503
    return result


@router.get("/circuit-breakers")
async def circuit_breaker_stats():
    """所有断路器状态"""
    registry = get_circuit_registry()
    return {"circuit_breakers": registry.get_all_stats()}
