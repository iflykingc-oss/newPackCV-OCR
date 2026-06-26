"""
LLM Provider管理API
查询可用的LLM Provider、模型列表、价格、状态
"""
import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/providers", tags=["providers"])

# 配置路径
CONFIG_PATH = os.path.join(
    os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects"),
    "data/llm_providers.json",
)

# 内存缓存（5分钟TTL）
_cache: Dict[str, Any] = {"data": None, "timestamp": 0}
CACHE_TTL = 300


def _load_providers() -> Dict[str, Any]:
    """加载LLM Provider配置（带缓存）"""
    now = time.time()
    if _cache["data"] is not None and (now - _cache["timestamp"]) < CACHE_TTL:
        return _cache["data"]

    if not os.path.exists(CONFIG_PATH):
        return {
            "providers": [],
            "models": [],
            "routing_rules": [],
            "error": f"config not found: {CONFIG_PATH}",
        }

    with open(CONFIG_PATH, "r", encoding="utf-8") as fd:
        data = json.load(fd)

    _cache["data"] = data
    _cache["timestamp"] = now
    return data


class ModelInfo(BaseModel):
    model_id: str
    provider: str
    display_name: str
    tier: str
    capabilities: List[str]
    pricing: Dict[str, float]
    context_window: int
    supports_json_mode: bool
    supports_vision: bool


class ProviderInfo(BaseModel):
    provider_id: str
    name: str
    enabled: bool
    api_base: str
    model_count: int
    circuit_breaker_state: str = "closed"
    last_error: Optional[str] = None
    success_rate_24h: float = 100.0


@router.get("")
async def list_providers():
    """
    列出所有LLM Provider（公开）
    返回Provider列表、可用模型、当前路由策略
    """
    data = _load_providers()
    providers = data.get("providers", [])
    models = data.get("models", [])

    # 计算每个provider的模型数
    model_count_map: Dict[str, int] = {}
    for m in models:
        provider = m.get("provider", "unknown")
        model_count_map[provider] = model_count_map.get(provider, 0) + 1

    result_providers = [
        ProviderInfo(
            provider_id=p.get("provider_id", ""),
            name=p.get("name", ""),
            enabled=p.get("enabled", False),
            api_base=p.get("api_base", ""),
            model_count=model_count_map.get(p.get("provider_id", ""), 0),
            circuit_breaker_state="closed",  # TODO: 从熔断器读取
            success_rate_24h=99.5,  # TODO: 从metrics读取
        )
        for p in providers
    ]

    return {
        "providers": [p.model_dump() for p in result_providers],
        "total_providers": len(result_providers),
        "total_models": len(models),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/models")
async def list_models(
    provider: Optional[str] = None,
    capability: Optional[str] = None,
    tier: Optional[str] = None,
):
    """
    列出所有可用模型（支持筛选）
    - provider: 按Provider筛选
    - capability: 按能力筛选（text_generation/vision/embedding）
    - tier: 按等级筛选（flagship/main/lite）
    """
    data = _load_providers()
    models = data.get("models", [])

    # 筛选
    filtered = models
    if provider:
        filtered = [m for m in filtered if m.get("provider") == provider]
    if capability:
        filtered = [m for m in filtered if capability in m.get("capabilities", [])]
    if tier:
        filtered = [m for m in filtered if m.get("tier") == tier]

    return {
        "models": filtered,
        "total": len(filtered),
        "filters": {"provider": provider, "capability": capability, "tier": tier},
    }


@router.get("/routing")
async def get_routing_strategy():
    """
    获取当前路由策略
    返回Fallback链、降级规则、限流阈值
    """
    data = _load_providers()
    return {
        "routing_rules": data.get("routing_rules", []),
        "fallback_chain": data.get("fallback_chain", []),
        "degradation_policies": data.get("degradation_policies", []),
        "rate_limit_policies": data.get("rate_limit_policies", {}),
    }


@router.get("/health")
async def providers_health():
    """
    Provider健康检查
    返回每个Provider的实时状态
    """
    data = _load_providers()
    providers = data.get("providers", [])

    health_status = []
    for p in providers:
        if not p.get("enabled", False):
            health_status.append({
                "provider_id": p.get("provider_id"),
                "healthy": False,
                "reason": "disabled",
            })
            continue

        # TODO: 实际 ping Provider API
        health_status.append({
            "provider_id": p.get("provider_id"),
            "healthy": True,
            "latency_ms": 120,  # 模拟
            "last_check": datetime.now(timezone.utc).isoformat(),
        })

    return {
        "providers": health_status,
        "total_healthy": sum(1 for h in health_status if h.get("healthy")),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/{provider_id}")
async def get_provider_detail(provider_id: str):
    """获取Provider详情"""
    data = _load_providers()
    providers = data.get("providers", [])

    provider = next((p for p in providers if p.get("provider_id") == provider_id), None)
    if not provider:
        raise HTTPException(status_code=404, detail=f"Provider not found: {provider_id}")

    # 关联的模型
    models = [m for m in data.get("models", []) if m.get("provider") == provider_id]

    return {
        **provider,
        "models": models,
        "model_count": len(models),
    }
