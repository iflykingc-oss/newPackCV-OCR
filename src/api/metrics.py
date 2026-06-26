"""
PackCV-OCR Prometheus 业务指标埋点
提供Counter/Gauge/Histogram三类指标，覆盖API调用/计费/限流/降级/审计/工作流
"""
import time
from typing import Optional
from contextlib import contextmanager

try:
    from prometheus_client import Counter, Gauge, Histogram, Summary, Info
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # 降级为None，避免依赖问题
    Counter = Gauge = Histogram = Summary = Info = None
    generate_latest = lambda: b""
    CONTENT_TYPE_LATEST = "text/plain"


# ============================================================
# API 业务指标
# ============================================================

API_REQUESTS_TOTAL = Counter(
    "packcv_api_requests_total",
    "API请求总数",
    ["tenant_id", "endpoint", "method", "status"]
) if PROMETHEUS_AVAILABLE else None

API_REQUEST_LATENCY = Histogram(
    "packcv_api_request_latency_seconds",
    "API请求延迟（秒）",
    ["endpoint", "method"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0)
) if PROMETHEUS_AVAILABLE else None

API_REQUEST_SIZE = Histogram(
    "packcv_api_request_size_bytes",
    "API请求大小（字节）",
    ["endpoint"]
) if PROMETHEUS_AVAILABLE else None

API_RESPONSE_SIZE = Histogram(
    "packcv_api_response_size_bytes",
    "API响应大小（字节）",
    ["endpoint"]
) if PROMETHEUS_AVAILABLE else None

# ============================================================
# 计费指标
# ============================================================

BILLING_TOKENS_TOTAL = Counter(
    "packcv_billing_tokens_total",
    "计费Token总数",
    ["tenant_id", "model", "type"]  # type=input/output
) if PROMETHEUS_AVAILABLE else None

BILLING_COST_USD_TOTAL = Counter(
    "packcv_billing_cost_usd_total",
    "计费成本总额（USD）",
    ["tenant_id", "model"]
) if PROMETHEUS_AVAILABLE else None

BILLING_INVOICES_TOTAL = Counter(
    "packcv_billing_invoices_total",
    "账单生成总数",
    ["tier", "status"]
) if PROMETHEUS_AVAILABLE else None

# ============================================================
# 限流指标
# ============================================================

RATE_LIMIT_HITS_TOTAL = Counter(
    "packcv_rate_limit_hits_total",
    "限流命中次数",
    ["tenant_id", "type"]  # type=rpm/tpm
) if PROMETHEUS_AVAILABLE else None

RATE_LIMIT_TENANTS_ACTIVE = Gauge(
    "packcv_rate_limit_tenants_active",
    "活跃租户数"
) if PROMETHEUS_AVAILABLE else None

# ============================================================
# 降级/熔断指标
# ============================================================

FALLBACK_TRIGGERS_TOTAL = Counter(
    "packcv_fallback_triggers_total",
    "模型降级触发次数",
    ["from_model", "to_model", "reason"]
) if PROMETHEUS_AVAILABLE else None

CIRCUIT_BREAKER_STATE = Gauge(
    "packcv_circuit_breaker_state",
    "熔断器状态（0=closed, 1=open, 2=half_open）",
    ["model"]
) if PROMETHEUS_AVAILABLE else None

# ============================================================
# 工作流指标
# ============================================================

WORKFLOW_EXECUTIONS_TOTAL = Counter(
    "packcv_workflow_executions_total",
    "工作流执行总数",
    ["scenario", "success"]
) if PROMETHEUS_AVAILABLE else None

WORKFLOW_LATENCY = Histogram(
    "packcv_workflow_latency_seconds",
    "工作流执行延迟（秒）",
    ["scenario"],
    buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0)
) if PROMETHEUS_AVAILABLE else None

WORKFLOW_NODE_LATENCY = Histogram(
    "packcv_workflow_node_latency_seconds",
    "工作流节点延迟（秒）",
    ["node_name"],
    buckets=(0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0)
) if PROMETHEUS_AVAILABLE else None

# ============================================================
# 数据脱敏/审计指标
# ============================================================

DATA_MASKING_TOTAL = Counter(
    "packcv_data_masking_total",
    "数据脱敏总数",
    ["type", "mode"]  # type=id_card/phone/bank_card/email/name, mode=partial/full
) if PROMETHEUS_AVAILABLE else None

AUDIT_LOGS_TOTAL = Counter(
    "packcv_audit_logs_total",
    "审计日志总数",
    ["action", "status"]
) if PROMETHEUS_AVAILABLE else None

# ============================================================
# 租户/系统指标
# ============================================================

TENANTS_TOTAL = Gauge(
    "packcv_tenants_total",
    "租户总数",
    ["tier", "status"]
) if PROMETHEUS_AVAILABLE else None

REDIS_CONNECTIONS = Gauge(
    "packcv_redis_connections_active",
    "Redis活跃连接数"
) if PROMETHEUS_AVAILABLE else None

POSTGRES_CONNECTIONS = Gauge(
    "packcv_postgres_connections_active",
    "PostgreSQL活跃连接数"
) if PROMETHEUS_AVAILABLE else None

# ============================================================
# 系统信息
# ============================================================

SYSTEM_INFO = Info(
    "packcv_system",
    "PackCV-OCR系统信息"
) if PROMETHEUS_AVAILABLE else None


# ============================================================
# 辅助函数
# ============================================================

def record_api_request(tenant_id: str, endpoint: str, method: str, status: int) -> None:
    """记录API请求"""
    if API_REQUESTS_TOTAL:
        API_REQUESTS_TOTAL.labels(
            tenant_id=tenant_id or "anonymous",
            endpoint=endpoint,
            method=method,
            status=str(status)
        ).inc()


def record_billing_tokens(tenant_id: str, model: str, input_tokens: int, output_tokens: int, cost_usd: float) -> None:
    """记录计费Token和成本"""
    if not PROMETHEUS_AVAILABLE:
        return
    BILLING_TOKENS_TOTAL.labels(tenant_id=tenant_id, model=model, type="input").inc(input_tokens)
    BILLING_TOKENS_TOTAL.labels(tenant_id=tenant_id, model=model, type="output").inc(output_tokens)
    BILLING_COST_USD_TOTAL.labels(tenant_id=tenant_id, model=model).inc(cost_usd)


def record_rate_limit_hit(tenant_id: str, limit_type: str) -> None:
    """记录限流命中"""
    if RATE_LIMIT_HITS_TOTAL:
        RATE_LIMIT_HITS_TOTAL.labels(tenant_id=tenant_id, type=limit_type).inc()


def record_fallback(from_model: str, to_model: str, reason: str) -> None:
    """记录模型降级"""
    if FALLBACK_TRIGGERS_TOTAL:
        FALLBACK_TRIGGERS_TOTAL.labels(
            from_model=from_model, to_model=to_model, reason=reason
        ).inc()


def record_workflow_execution(scenario: str, success: bool, latency_seconds: float) -> None:
    """记录工作流执行"""
    if not PROMETHEUS_AVAILABLE:
        return
    WORKFLOW_EXECUTIONS_TOTAL.labels(
        scenario=scenario or "unknown",
        success=str(success).lower()
    ).inc()
    WORKFLOW_LATENCY.labels(scenario=scenario or "unknown").observe(latency_seconds)


def record_data_masking(masking_type: str, mode: str) -> None:
    """记录数据脱敏"""
    if DATA_MASKING_TOTAL:
        DATA_MASKING_TOTAL.labels(type=masking_type, mode=mode).inc()


def record_audit(action: str, status: str = "success") -> None:
    """记录审计"""
    if AUDIT_LOGS_TOTAL:
        AUDIT_LOGS_TOTAL.labels(action=action, status=status).inc()


@contextmanager
def measure_latency(metric, **labels):
    """延迟测量上下文管理器"""
    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        if metric and PROMETHEUS_AVAILABLE:
            metric.labels(**labels).observe(elapsed)


def get_metrics() -> bytes:
    """获取Prometheus格式的指标数据"""
    if PROMETHEUS_AVAILABLE:
        return generate_latest()
    return b""


def set_system_info(version: str, environment: str) -> None:
    """设置系统信息"""
    if SYSTEM_INFO:
        SYSTEM_INFO.info({
            "version": version,
            "environment": environment,
            "python_version": "3.11",
            "framework": "FastAPI + LangGraph"
        })


# 初始化
set_system_info(version="7.0-prod", environment="production")
