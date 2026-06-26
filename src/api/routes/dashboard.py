"""
管理Dashboard API
总览系统状态：租户数、API调用、计费、限流、错误率
"""
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

from fastapi import APIRouter

from tenancy.api_key_manager import APIKeyManager
from utils.redis_client import redis_client

router = APIRouter(prefix="/admin/dashboard", tags=["dashboard"])


def _safe_list_tenants() -> List[Dict[str, Any]]:
    """安全获取租户列表（兼容不同实现）"""
    try:
        if hasattr(APIKeyManager, "list_tenants"):
            return APIKeyManager.list_tenants() or []
    except Exception:
        pass
    return []


@router.get("")
async def admin_dashboard():
    """
    管理员Dashboard（公开仅供演示，生产需鉴权）
    返回系统总览数据
    """
    # 1. 租户统计
    all_tenants = _safe_list_tenants()
    tier_distribution: Dict[str, int] = {}
    active_count = 0
    for t in all_tenants:
        tier = t.get("tier", "unknown") if isinstance(t, dict) else "unknown"
        tier_distribution[tier] = tier_distribution.get(tier, 0) + 1
        if isinstance(t, dict) and t.get("active", True):
            active_count += 1

    # 2. 当前月份
    now = datetime.now(timezone.utc)
    year_month = now.strftime("%Y-%m")

    # 3. 总调用数（从Redis聚合）
    total_calls = 0
    total_tokens = 0
    for t in all_tenants:
        if not isinstance(t, dict):
            continue
        tid = t.get("tenant_id") or t.get("id")
        if not tid:
            continue
        try:
            key = f"billing:usage:{tid}:{year_month}"
            usage = redis_client.client.hgetall(key) or {}
            if usage:
                total_calls += int(usage.get("call_count", 0) or 0)
                total_tokens += int(usage.get("total_tokens", 0) or 0)
        except Exception:
            pass

    # 4. 错误率（从Prometheus指标读取）
    error_rate = 0.0
    p95_latency = 0.0
    try:
        metrics_text = ""
        from api.metrics import ERROR_COUNTER, REQUEST_LATENCY
        # 简化：直接返回 0，详情从 /metrics 端点拿
    except Exception:
        pass

    # 5. 系统状态
    system_status = {
        "redis": redis_client.health_check() if hasattr(redis_client, "health_check") else "unknown",
        "env": os.getenv("ENV", "dev"),
        "version": "7.0.0",
        "uptime_seconds": int(time.time()),
    }

    return {
        "generated_at": now.isoformat(),
        "tenants": {
            "total": len(all_tenants),
            "active": active_count,
            "by_tier": tier_distribution,
        },
        "usage_this_month": {
            "total_calls": total_calls,
            "total_tokens": total_tokens,
            "period": year_month,
        },
        "errors": {
            "error_rate": error_rate,
            "p95_latency_ms": p95_latency,
        },
        "system": system_status,
    }


@router.get("/tenants")
async def dashboard_tenants():
    """Dashboard - 租户列表"""
    return {"tenants": _safe_list_tenants(), "count": len(_safe_list_tenants())}


@router.get("/health")
async def dashboard_health():
    """Dashboard - 深度健康检查"""
    checks: Dict[str, Any] = {}
    # Redis
    try:
        checks["redis"] = "ok" if redis_client.health_check() else "degraded"
    except Exception as e:
        checks["redis"] = f"error: {e}"
    # LLM Providers（仅判断配置存在）
    try:
        from utils.llm_client import list_providers
        providers = list_providers() if hasattr(list_providers, "__call__") else []
        checks["llm_providers"] = {"count": len(providers), "status": "ok"}
    except Exception as e:
        checks["llm_providers"] = {"status": f"degraded: {e}"}

    overall = "ok" if all(v == "ok" or (isinstance(v, dict) and v.get("status") == "ok") for v in checks.values()) else "degraded"
    return {"overall": overall, "checks": checks}
