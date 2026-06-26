"""GraphQL Schema — PackCV 统一查询接口

基于 Ariadne GraphQL，提供：
- 租户信息查询
- 场景/配额/计费 统一查询
- 断路器/灰度/缓存 运维查询
- 健康状态
"""
import json
import os
import time
from ariadne import QueryType, make_executable_schema, graphql_sync, ObjectType
from ariadne.asgi import GraphQL
from fastapi import Request, Response

# ==================== SDL 定义 ====================

type_defs = """
    type Query {
        tenant(apiKey: String!): TenantInfo
        scenarios: [ScenarioItem!]!
        cacheStats: CacheStats!
        circuitBreakers: [CircuitBreakerStatus!]!
        canaries: [CanaryInfo!]!
        fewShotScenarios: [FewShotScenario!]!
        providers: [ProviderInfo!]!
        health: HealthStatus!
    }

    type TenantInfo {
        tenantId: String!
        tenantName: String!
        tier: String!
        isolationLevel: String!
        quota: QuotaInfo!
        allowedModels: [String!]!
    }

    type QuotaInfo {
        baseRpm: Int!
        baseTpm: Int!
        baseConcurrent: Int!
        monthlyTokenQuota: Int!
        monthlyCallQuota: Int!
        monthlyTokensUsed: Int!
        monthlyCallsUsed: Int!
        billingMode: String!
    }

    type ScenarioItem {
        name: String!
        description: String!
        version: String!
        enabled: Boolean!
    }

    type CacheStats {
        size: Int!
        maxSize: Int!
        hits: Int!
        misses: Int!
        hitRate: Float!
        totalRequests: Int!
    }

    type CircuitBreakerStatus {
        provider: String!
        state: String!
        failureCount: Int!
        lastFailureTime: Float
    }

    type CanaryInfo {
        canaryId: String!
        name: String!
        version: String!
        percentage: Int!
        enabled: Boolean!
    }

    type FewShotScenario {
        scenario: String!
        count: Int!
    }

    type ProviderInfo {
        id: String!
        name: String!
        models: [String!]!
        enabled: Boolean!
    }

    type HealthStatus {
        live: Boolean!
        ready: Boolean!
        startup: Boolean!
        uptimeSeconds: Float!
    }
"""

# ==================== Resolvers ====================

query = QueryType()


@query.field("tenant")
def resolve_tenant(_, info, apiKey):
    from tenancy.api_key_manager import APIKeyManager
    t = APIKeyManager.verify_api_key_only(apiKey)
    if t is None:
        return None
    q = t.quota
    is_dict = isinstance(q, dict)
    return {
        "tenantId": t.tenant_id,
        "tenantName": t.tenant_name,
        "tier": t.tier if isinstance(t.tier, str) else t.tier.value,
        "isolationLevel": t.isolation_level if isinstance(t.isolation_level, str) else t.isolation_level.value,
        "quota": {
            "baseRpm": q.get("base_rpm", 0) if is_dict else getattr(q, "base_rpm", 0),
            "baseTpm": q.get("base_tpm", 0) if is_dict else getattr(q, "base_tpm", 0),
            "baseConcurrent": q.get("base_concurrent", 0) if is_dict else getattr(q, "base_concurrent", 0),
            "monthlyTokenQuota": q.get("monthly_token_quota", 0) if is_dict else getattr(q, "monthly_token_quota", 0),
            "monthlyCallQuota": q.get("monthly_call_quota", 0) if is_dict else getattr(q, "monthly_call_quota", 0),
            "monthlyTokensUsed": q.get("monthly_tokens_used", 0) if is_dict else getattr(q, "monthly_tokens_used", 0),
            "monthlyCallsUsed": q.get("monthly_calls_used", 0) if is_dict else getattr(q, "monthly_calls_used", 0),
            "billingMode": q.get("billing_mode", "hybrid") if is_dict else getattr(q, "billing_mode", "hybrid"),
        },
        "allowedModels": t.allowed_models if isinstance(t.allowed_models, list) else [],
    }


@query.field("scenarios")
def resolve_scenarios(_, info):
    from api.routes.system import SCENARIO_REGISTRY
    result = []
    if isinstance(SCENARIO_REGISTRY, dict):
        for name, s in SCENARIO_REGISTRY.items():
            if isinstance(s, dict):
                result.append({
                    "name": name,
                    "description": s.get("description", ""),
                    "version": s.get("version", "1.0"),
                    "enabled": s.get("enabled", True),
                })
    return result


@query.field("cacheStats")
def resolve_cache_stats(_, info):
    from intelligence.llm_cache import LLMResponseCache
    cache = LLMResponseCache()
    s = cache.stats()
    return {
        "size": s.get("size", 0),
        "maxSize": s.get("max_size", 0),
        "hits": s.get("hits", 0),
        "misses": s.get("misses", 0),
        "hitRate": s.get("hit_rate", 0.0),
        "totalRequests": s.get("total_requests", 0),
    }


@query.field("circuitBreakers")
def resolve_circuit_breakers(_, info):
    from resilience.circuit_breaker import CircuitBreakerRegistry
    result = []
    for name, cb in CircuitBreakerRegistry._breakers.items():
        result.append({
            "provider": name,
            "state": cb.state.value if hasattr(cb.state, "value") else str(cb.state),
            "failureCount": cb.failure_count,
            "lastFailureTime": cb.last_failure_time,
        })
    return result


@query.field("canaries")
def resolve_canaries(_, info):
    from gradual_rollout import CanaryDeployer
    deployer = CanaryDeployer()
    canaries = deployer.list_canaries()
    return [
        {
            "canaryId": c.canary_id,
            "name": c.name,
            "version": c.version,
            "percentage": c.percentage,
            "enabled": c.enabled,
        }
        for c in canaries
    ]


@query.field("fewShotScenarios")
def resolve_few_shot_scenarios(_, info):
    from intelligence.few_shot import FewShotManager
    fsm = FewShotManager()
    scenarios = fsm.list_scenarios()
    return [
        {"scenario": s.get("scenario", ""), "count": s.get("count", 0)}
        for s in scenarios
    ]


@query.field("providers")
def resolve_providers(_, info):
    providers_path = os.path.join(
        os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects"),
        "data", "llm_providers.json"
    )
    result = []
    try:
        with open(providers_path, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            for p in data:
                result.append({
                    "id": p.get("id", ""),
                    "name": p.get("name", ""),
                    "models": p.get("models", []),
                    "enabled": p.get("enabled", True),
                })
    except Exception:
        pass
    return result


@query.field("health")
def resolve_health(_, info):
    startup_file = os.path.join("/tmp", "packcv-startup-time")
    uptime = 0.0
    try:
        with open(startup_file, "r") as f:
            uptime = time.time() - float(f.read().strip())
    except Exception:
        uptime = 0.0
    return {
        "live": True,
        "ready": True,
        "startup": True,
        "uptimeSeconds": uptime,
    }


# ==================== Schema + ASGI App ====================

schema = make_executable_schema(type_defs, query)
graphql_app = GraphQL(schema)
