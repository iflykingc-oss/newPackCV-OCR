"""Phase 5 API 路由 - 智能增强 + 企业级 + 国际化"""
import os
from typing import Optional, List
from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel, Field

from intelligence import (
    LLMResponseCache, FewShotManager, ABTestFramework, ExperimentConfig, Variant
)
from auth_sso import RBACManager, Role, Permission, OIDCProvider
from gradual_rollout import CanaryDeployer
from streaming import get_sse_manager
import i18n as i18n_mod
from i18n import _

router = APIRouter(prefix="", tags=["Phase 5 智能增强+企业级"])

# 全局单例
_cache = LLMResponseCache()
_fsm = FewShotManager()
_rbac = RBACManager()
_canary = CanaryDeployer()
_sso_providers = {}

# ===================== 缓存 =====================
@router.get("/intelligence/cache/stats")
async def cache_stats():
    return _cache.stats()

class CacheSetReq(BaseModel):
    key: str
    value: dict
    ttl: int = 3600

@router.post("/intelligence/cache/set")
async def cache_set(req: CacheSetReq):
    _cache.set(req.key, req.value, ttl=req.ttl)
    return {"status": "ok", "key": req.key, "ttl": req.ttl}

@router.get("/intelligence/cache/{key}")
async def cache_get(key: str):
    val = _cache.get(key)
    if val is None:
        raise HTTPException(404, "key not found")
    return {"key": key, "value": val}

# ===================== Few-shot =====================
@router.get("/intelligence/few-shot/stats")
async def few_shot_stats():
    """全部场景统计"""
    return _fsm.stats()

@router.get("/intelligence/few-shot/scenarios")
async def few_shot_scenarios():
    """场景列表及样本数"""
    return {"scenarios": _fsm.list_scenarios()}

class FewShotReq(BaseModel):
    scenario: str
    input_text: str
    output_text: str
    score: float = 1.0
    tags: List[str] = Field(default_factory=list)

@router.post("/intelligence/few-shot/add")
async def few_shot_add(req: FewShotReq):
    ex = _fsm.add_example(
        scenario=req.scenario,
        input_summary=req.input_text,
        output=req.output_text,
        score=req.score,
        tags=req.tags or None,
    )
    return {"example_id": ex.example_id, "scenario": ex.scenario, "score": ex.score}

@router.get("/intelligence/few-shot/list/{scenario}")
async def few_shot_list(scenario: str, k: int = Query(5, ge=1, le=20)):
    examples = _fsm.get_examples(scenario, top_k=k)
    return {"scenario": scenario, "count": len(examples), "examples": [
        {"example_id": e.example_id, "input": e.input_summary, "output": e.output, "score": e.score}
        for e in examples
    ]}

# ===================== A/B 测试 =====================
_ab_frameworks: dict = {}

class ABExperimentReq(BaseModel):
    name: str
    description: str = ""
    variants: List[dict] = Field(..., min_length=2, max_length=4)

@router.post("/intelligence/ab/create")
async def ab_create(req: ABExperimentReq):
    variants = [Variant(name=v["name"], weight=v.get("weight", 50)) for v in req.variants]
    exp = ExperimentConfig(name=req.name, description=req.description, variants=variants)
    _ab_frameworks[req.name] = ABTestFramework(exp)
    return {"experiment": req.name, "variants": [{"name": v.name, "weight": v.weight} for v in variants]}

@router.get("/intelligence/ab/{name}/assign")
async def ab_assign(name: str, user_id: str):
    if name not in _ab_frameworks:
        raise HTTPException(404, "experiment not found")
    frm = _ab_frameworks[name]
    variant = frm.assign(user_id)
    return {"experiment": name, "user_id": user_id, "variant": variant}

class ABRecordReq(BaseModel):
    user_id: str
    success: bool

@router.post("/intelligence/ab/{name}/record")
async def ab_record(name: str, req: ABRecordReq):
    if name not in _ab_frameworks:
        raise HTTPException(404, "experiment not found")
    _ab_frameworks[name].record(req.user_id, req.success)
    return {"experiment": name, "user_id": req.user_id, "recorded": True}

@router.get("/intelligence/ab/{name}/report")
async def ab_report(name: str):
    if name not in _ab_frameworks:
        raise HTTPException(404, "experiment not found")
    return _ab_frameworks[name].report()

# ===================== RBAC =====================
@router.get("/rbac/roles")
async def rbac_roles(tenant_id: Optional[str] = Query(None)):
    roles = _rbac.list_roles(tenant_id=tenant_id)
    return {"roles": [
        {
            "role_id": r.role_id,
            "name": r.name,
            "description": r.description,
            "permissions": [p.value if hasattr(p, "value") else str(p) for p in r.permissions],
            "tenant_id": r.tenant_id,
        }
        for r in roles
    ]}

class RoleReq(BaseModel):
    name: str
    permissions: List[str] = Field(default_factory=list)

@router.post("/rbac/role")
async def rbac_create(req: RoleReq):
    role = _rbac.create_role(req.name, permissions=[Permission(p) for p in req.permissions])
    return {"role": role.name, "permissions": [p.value for p in role.permissions]}

class CheckReq(BaseModel):
    user_id: str
    role: str  # 仅用于日志/审计
    permission: str
    tenant_id: Optional[str] = None

@router.post("/rbac/check")
async def rbac_check(req: CheckReq):
    # RBACManager.check_permission 是 user_id+perm 维度，role 信息写入响应
    has = _rbac.check_permission(req.user_id, Permission(req.permission), tenant_id=req.tenant_id)
    return {"user_id": req.user_id, "role": req.role, "permission": req.permission, "allowed": has, "tenant_id": req.tenant_id}

# ===================== SSO =====================
@router.get("/sso/providers")
async def sso_providers():
    return {"providers": list(_sso_providers.keys())}

class OIDCReq(BaseModel):
    name: str
    issuer: str
    client_id: str
    client_secret: str
    redirect_uri: str

@router.post("/sso/oidc")
async def sso_register(req: OIDCReq):
    provider = OIDCProvider(req.issuer, req.client_id, req.client_secret, req.redirect_uri)
    _sso_providers[req.name] = provider
    return {"name": req.name, "issuer": req.issuer, "client_id": req.client_id}

@router.get("/sso/{name}/authorize")
async def sso_authorize(name: str, state: str = "default"):
    if name not in _sso_providers:
        raise HTTPException(404, "provider not found")
    url = _sso_providers[name].get_authorization_url(state)
    return {"name": name, "state": state, "url": url}

# ===================== 灰度发布 =====================
@router.get("/canary/list")
async def canary_list():
    cn_list = _canary.list_canaries()
    return {"canaries": [
        {"canary_id": c.canary_id, "name": c.name, "version": c.version, "percentage": c.percentage}
        for c in cn_list
    ]}

class CanaryReq(BaseModel):
    name: str
    version: str
    percentage: int = Field(..., ge=0, le=100)
    whitelist: List[str] = Field(default_factory=list)

@router.post("/canary/create")
async def canary_create(req: CanaryReq):
    whitelist = set(req.whitelist) if req.whitelist else None
    cn = _canary.create_canary(
        name=req.name, version=req.version, percentage=req.percentage,
        whitelist=whitelist,
    )
    return {"canary_id": cn.canary_id, "name": cn.name, "version": cn.version, "percentage": cn.percentage}

@router.get("/canary/{name}/route")
async def canary_route(name: str, user_id: str, region: Optional[str] = None):
    """基于 user_id 决定是否命中灰度版本"""
    from gradual_rollout import RolloutStrategy
    # 找到对应 canary 配置
    target = None
    for c in _canary.list_canaries():
        if c.name == name:
            target = c
            break
    if target is None:
        raise HTTPException(404, "canary not found")
    if not target.enabled:
        return {"canary": name, "user_id": user_id, "version": None, "hit": False}
    # 简单按 user_id hash 取模
    if user_id in (target.whitelist or set()):
        return {"canary": name, "user_id": user_id, "version": target.version, "hit": True}
    if target.strategy == RolloutStrategy.PERCENTAGE:
        h = abs(hash(user_id)) % 100
        hit = h < target.percentage
    elif target.strategy == RolloutStrategy.HEADER and region:
        hit = region in (target.regions or set())
    else:
        hit = False
    return {"canary": name, "user_id": user_id, "version": target.version if hit else None, "hit": hit}

# ===================== SSE 流式 =====================
@router.get("/streaming/stats")
async def streaming_stats():
    return get_sse_manager().get_stats()

class StreamCreateReq(BaseModel):
    tag: Optional[str] = None

@router.post("/streaming/create")
async def streaming_create(req: StreamCreateReq):
    s = get_sse_manager().create_stream(tag=req.tag)
    return {
        "event_id": s.event_id,
        "tag": req.tag,
        "closed": s.closed,
    }

# ===================== i18n 国际化 =====================
@router.get("/i18n/{locale}/{key}")
async def i18n_translate(locale: str, key: str):
    i18n_mod.set_locale(locale)
    return {"locale": locale, "key": key, "value": _(key)}

@router.get("/i18n/locales")
async def i18n_locales():
    return {"locales": list(i18n_mod.SUPPORTED_LOCALES)}
