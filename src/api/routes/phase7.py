#!/usr/bin/env python3
"""Phase 7 功能优化 API 路由
端点:
- /docs-enhanced/* : API 文档增强
- /errors/* : 错误码管理
- /config/* : 配置热更新
- /lineage/* : 数据血缘
- /tracing/* : 分布式追踪
"""
from fastapi import APIRouter, Query, HTTPException, Body
from typing import Dict, Any, Optional, List

from api_docs import get_error_codes, get_api_examples, get_api_history, get_sdk_guide
from errors import ErrorRegistry, ErrorResponse
from config_hotreload import HotReloadManager, update_rate_limits, enable_provider, ConfigChange
from data_lineage import LineageStore, lineage_stats, DataLineage

router = APIRouter()


# ========== API 文档增强 ==========
@router.get("/docs-enhanced/error-codes", summary="获取错误码表")
async def api_error_codes(locale: str = Query(default="zh", description="语言 (zh/en/ja)")):
    """获取标准化错误码表"""
    return get_error_codes(locale)


@router.get("/docs-enhanced/examples", summary="获取 API 示例")
async def api_examples():
    """获取各 API 端点的请求/响应示例"""
    return get_api_examples()


@router.get("/docs-enhanced/history", summary="获取 API 变更历史")
async def api_history():
    """获取 API 版本变更历史"""
    return get_api_history()


@router.get("/docs-enhanced/sdk-guide", summary="获取 SDK 使用指南")
async def api_sdk_guide():
    """获取 Python/JavaScript SDK 使用指南"""
    return get_sdk_guide()


# ========== 错误码管理 ==========
@router.get("/errors/list", summary="列出所有错误码")
async def errors_list(locale: str = Query(default="zh")):
    """列出所有错误码及其说明"""
    return {"total": len(ErrorRegistry.list_all(locale)), "errors": ErrorRegistry.list_all(locale)}


@router.post("/errors/register", summary="注册新错误码")
async def errors_register(
    error_code: int = Body(..., description="错误码"),
    code: str = Body(..., description="错误类型码"),
    messages: Dict[str, str] = Body(..., description="多语言消息 {zh, en, ja}"),
    recovery_hint: Optional[str] = Body(default=None, description="恢复建议")
):
    """注册新的错误码"""
    ErrorRegistry.register_error(error_code, code, messages, recovery_hint)
    return {"success": True, "error_code": error_code}


# ========== 配置热更新 ==========
@router.get("/config/list", summary="列出所有配置")
async def config_list():
    """列出所有已加载配置及其版本"""
    return HotReloadManager.list_configs()


@router.get("/config/{config_name}", summary="获取配置内容")
async def config_get(config_name: str):
    """获取指定配置内容"""
    config = HotReloadManager.get_config(config_name)
    if not config:
        raise HTTPException(status_code=404, detail=f"Config {config_name} not found")
    return {"config_name": config_name, "version": HotReloadManager.get_version(config_name), "content": config}


@router.post("/config/{config_name}/reload", summary="重新加载配置")
async def config_reload(config_name: str):
    """从磁盘重新加载配置"""
    config = HotReloadManager.get_config(config_name, reload=True)
    return {"success": True, "config_name": config_name, "version": HotReloadManager.get_version(config_name)}


@router.post("/config/rate-limits/update", summary="更新限流阈值")
async def config_rate_limits_update(
    tenant_tier: str = Body(...),
    rpm: Optional[int] = Body(default=None),
    tpm: Optional[int] = Body(default=None),
    concurrent: Optional[int] = Body(default=None)
):
    """动态更新租户等级的限流阈值"""
    change = update_rate_limits(tenant_tier, rpm, tpm, concurrent)
    return {"success": True, "change": change.model_dump()}


@router.post("/config/provider/toggle", summary="启用/禁用 Provider")
async def config_provider_toggle(
    provider_id: str = Body(...),
    enabled: bool = Body(default=True)
):
    """启用或禁用指定的 LLM Provider"""
    change = enable_provider(provider_id, enabled)
    return {"success": True, "provider_id": provider_id, "enabled": enabled, "change": change.model_dump()}


# ========== 数据血缘 ==========
@router.get("/lineage/stats", summary="血缘统计")
async def lineage_statistics():
    """获取数据血缘统计信息"""
    return lineage_stats()


@router.get("/lineage/{lineage_id}", summary="获取血缘详情")
async def lineage_get(lineage_id: str):
    """获取指定血缘记录详情"""
    lineage = LineageStore.get(lineage_id)
    if not lineage:
        raise HTTPException(status_code=404, detail=f"Lineage {lineage_id} not found")
    return lineage.model_dump()


@router.get("/lineage/by-run/{run_id}", summary="按运行ID获取血缘")
async def lineage_by_run(run_id: str):
    """按工作流运行ID获取血缘"""
    lineage = LineageStore.get_by_run(run_id)
    if not lineage:
        raise HTTPException(status_code=404, detail=f"Lineage for run {run_id} not found")
    return lineage.model_dump()


@router.get("/lineage/by-tenant/{tenant_id}", summary="按租户列出血缘")
async def lineage_by_tenant(tenant_id: str, limit: int = Query(default=50)):
    """按租户列出血缘记录"""
    lineages = LineageStore.list_by_tenant(tenant_id, limit)
    return {"total": len(lineages), "tenant_id": tenant_id, "lineages": [l.model_dump() for l in lineages]}


@router.post("/lineage/search", summary="搜索血缘")
async def lineage_search(query: Dict[str, Any] = Body(default={})):
    """按条件搜索血缘记录"""
    results = LineageStore.search(query)
    return {"total": len(results), "results": [r.model_dump() for r in results]}


# ========== 分布式追踪状态 ==========
@router.get("/tracing/status", summary="追踪状态")
async def tracing_status():
    """获取 OpenTelemetry 追踪状态"""
    from tracing import get_tracer
    tracer = get_tracer()
    return {
        "enabled": tracer is not None,
        "service_name": getattr(tracer, "_service_name", "packcv-api") if tracer else None
    }