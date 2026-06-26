"""API 版本管理 + 废弃策略 路由"""
from fastapi import APIRouter, Query, HTTPException
from typing import Optional

from api_versioning import version_manager, DeprecationLevel, init_default_endpoints

router = APIRouter(prefix="/api-lifecycle", tags=["API 版本管理"])

# 初始化默认端点
init_default_endpoints()


@router.get("/endpoints")
async def list_endpoints(
    version: Optional[str] = Query(None, description="按版本过滤"),
    level: Optional[str] = Query(None, description="按废弃级别过滤(active/deprecated/sunset/removed)"),
):
    """列出所有端点及其废弃状态"""
    dep_level = None
    if level:
        try:
            dep_level = DeprecationLevel(level)
        except ValueError:
            raise HTTPException(400, f"无效废弃级别: {level}")
    endpoints = version_manager.list_endpoints(version=version, level=dep_level)
    return {"count": len(endpoints), "endpoints": [
        {
            "path": e.path,
            "method": e.method,
            "version": e.version,
            "deprecation_level": e.deprecation_level.value,
            "deprecated_since": e.deprecated_since,
            "sunset_date": e.sunset_date,
            "alternate_path": e.alternate_path,
            "description": e.description,
            "migration_guide": e.migration_guide,
        }
        for e in endpoints
    ]}


@router.get("/stats")
async def version_stats():
    """端点废弃统计"""
    return {"stats": version_manager.stats(), "current_version": version_manager.get_current_version()}


@router.get("/check")
async def check_endpoint(path: str = Query(...), method: str = Query("GET")):
    """检查特定端点的废弃状态"""
    info = version_manager.check(path, method.upper())
    if info is None:
        raise HTTPException(404, "端点未注册")
    headers = version_manager.get_deprecation_headers(path, method.upper())
    return {
        "path": info.path,
        "method": info.method,
        "version": info.version,
        "deprecation_level": info.deprecation_level.value,
        "deprecated_since": info.deprecated_since,
        "sunset_date": info.sunset_date,
        "alternate_path": info.alternate_path,
        "migration_guide": info.migration_guide,
        "response_headers": headers,
    }


class DeprecateReq:
    """废弃请求 — 用 dict 接收"""
    pass

from pydantic import BaseModel, Field
from typing import List as TList


class DeprecateEndpointReq(BaseModel):
    path: str
    method: str = "GET"
    alternate_path: Optional[str] = None
    sunset_date: Optional[str] = None
    migration_guide: str = ""


@router.post("/deprecate")
async def deprecate_endpoint(req: DeprecateEndpointReq):
    """标记端点为废弃"""
    info = version_manager.deprecate(
        path=req.path,
        method=req.method,
        alternate_path=req.alternate_path,
        sunset_date=req.sunset_date,
        migration_guide=req.migration_guide,
    )
    if info is None:
        raise HTTPException(404, f"端点 {req.method} {req.path} 未注册")
    return {
        "path": info.path,
        "method": info.method,
        "deprecation_level": info.deprecation_level.value,
        "sunset_date": info.sunset_date,
        "alternate_path": info.alternate_path,
    }


class RegisterEndpointReq(BaseModel):
    path: str
    method: str = "GET"
    version: str = "v1"
    description: str = ""


@router.post("/register")
async def register_endpoint(req: RegisterEndpointReq):
    """注册端点"""
    info = version_manager.register(
        path=req.path, method=req.method,
        version=req.version, description=req.description,
    )
    return {
        "path": info.path,
        "method": info.method,
        "version": info.version,
        "deprecation_level": info.deprecation_level.value,
    }


@router.get("/versions")
async def list_versions():
    """版本列表 + 别名"""
    versions = set()
    aliases = dict(version_manager._version_aliases)
    for ep in version_manager.list_endpoints():
        versions.add(ep.version)
    return {"versions": sorted(versions), "aliases": aliases, "current": version_manager.get_current_version()}
