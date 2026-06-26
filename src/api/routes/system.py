"""健康检查 + 租户管理路由"""

from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from tenancy.api_key_manager import APIKeyManager
from tenancy.models import TenantModel, TenantTier, UsageStats


router = APIRouter(prefix="/api/v1", tags=["system"])


# ============= 健康检查 =============

@router.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "ok",
        "service": "packcv-ocr",
        "version": "v7.0-prod",
        "timestamp": datetime.now().isoformat(),
    }


# ============= 租户管理（管理员）=============

class CreateTenantRequest(BaseModel):
    """创建租户请求"""
    tenant_name: str = Field(..., min_length=1, max_length=100)
    tier: str = Field(default="free", description="租户等级")
    contact_email: str = Field(..., description="联系邮箱")
    contact_name: Optional[str] = None
    env: str = Field(default="test", description="环境: live/test")


class CreateTenantResponse(BaseModel):
    """创建租户响应"""
    tenant_id: str
    tenant_name: str
    tier: str
    api_key: str
    api_secret: str = Field(..., description="⚠️ 仅创建时返回一次，请妥善保存")
    redis_namespace: str
    allowed_models: List[str]


@router.post("/admin/tenants", response_model=CreateTenantResponse)
async def create_tenant(req: CreateTenantRequest):
    """创建新租户（管理员接口）

    注意：API Secret仅在创建时返回一次，请妥善保存！
    """
    try:
        tier_enum = TenantTier(req.tier)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"无效的tier，可选: {[t.value for t in TenantTier]}",
        )

    tenant, api_secret = APIKeyManager.create_tenant(
        tenant_name=req.tenant_name,
        tier=tier_enum,
        contact_email=req.contact_email,
        env=req.env,
    )
    return CreateTenantResponse(
        tenant_id=tenant.tenant_id,
        tenant_name=tenant.tenant_name,
        tier=tenant.tier if isinstance(tenant.tier, str) else tenant.tier.value,
        api_key=tenant.api_key,
        api_secret=api_secret,
        redis_namespace=tenant.redis_namespace,
        allowed_models=tenant.allowed_models,
    )


@router.post("/admin/tenants/demo")
async def setup_demo_tenants():
    """初始化演示租户（开发环境用）"""
    APIKeyManager.setup_demo_tenants()
    tenants = APIKeyManager.list_tenants()
    return {
        "message": "演示租户已创建",
        "count": len(tenants),
        "tenants": [
            {
                "tenant_id": t.tenant_id,
                "tenant_name": t.tenant_name,
                "tier": t.tier if isinstance(t.tier, str) else t.tier.value,
                "api_key": t.api_key,
            }
            for t in tenants
        ],
    }


@router.get("/admin/tenants", response_model=List[TenantModel])
async def list_tenants():
    """列出所有租户"""
    return APIKeyManager.list_tenants()


@router.delete("/admin/tenants/{api_key}")
async def revoke_tenant(api_key: str):
    """撤销租户"""
    success = APIKeyManager.revoke_tenant(api_key)
    if not success:
        raise HTTPException(status_code=404, detail="租户不存在")
    return {"message": "租户已撤销", "api_key": api_key}


# ============= 租户自服务 =============

@router.get("/me")
async def get_current_tenant_info():
    """获取当前租户信息（需要鉴权）"""
    from tenancy.context import TenantContext

    ctx = TenantContext.require()
    return {
        "tenant_id": ctx["tenant_id"],
        "tenant_name": ctx["tenant_name"],
        "tier": ctx["tier"],
        "isolation_level": ctx["isolation_level"],
        "request_id": ctx["request_id"],
        "quota": ctx["quota"],
        "allowed_models": ctx["allowed_models"],
    }


@router.get("/usage", response_model=UsageStats)
async def get_usage_stats(year_month: Optional[str] = None):
    """获取使用统计"""
    from tenancy.context import TenantContext

    ctx = TenantContext.require()
    tenant_id = ctx["tenant_id"]
    period = year_month or datetime.now().strftime("%Y%m")

    from utils.redis_client import redis_client

    usage = redis_client.hgetall(tenant_id, f"usage:{period}")
    return UsageStats(
        tenant_id=tenant_id,
        period=period,
        total_calls=int(usage.get("calls", 0)),
        total_tokens=int(usage.get("tokens", 0)),
        total_cost_usd=float(usage.get("cost", 0.0)),
        by_model={},
        by_scenario={},
    )
