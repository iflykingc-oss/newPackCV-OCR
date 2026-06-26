#!/usr/bin/env python3
"""分布式追踪模块 — OpenTelemetry 集成
功能:
- 自动追踪 FastAPI 请求
- LLM 调用追踪
- Redis 操作追踪
- 自定义 Span 标注
- 导出 OTLP / Jaeger / Console
"""
import os
import time
import logging
from typing import Optional, Dict, Any, Callable
from functools import wraps

# OpenTelemetry 是可选依赖;未安装时使用 no-op stub,保证系统可启动
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.trace import Status, StatusCode, Span
    _OPENTELEMETRY_AVAILABLE = True
except ImportError:  # noqa: BLE001
    _OPENTELEMETRY_AVAILABLE = False
    logger_fallback = logging.getLogger(__name__ + ".fallback")
    logger_fallback.info("opentelemetry not installed, using no-op tracer stub")
    # No-op stub 类型
    Status = None  # type: ignore
    StatusCode = None  # type: ignore
    Span = None  # type: ignore
    TracerProvider = None  # type: ignore
    BatchSpanProcessor = None  # type: ignore
    ConsoleSpanExporter = None  # type: ignore
    Resource = None  # type: ignore
    SERVICE_NAME = "service.name"  # type: ignore
    SERVICE_VERSION = "service.version"  # type: ignore
    FastAPIInstrumentor = None  # type: ignore
    trace = None  # type: ignore

logger = logging.getLogger(__name__)

# 全局 tracer
_tracer: Optional[Any] = None
_provider: Optional[Any] = None


def init_tracing(
    service_name: str = "packcv-api",
    service_version: str = "7.0.0",
    exporter_type: str = "console",
    otlp_endpoint: Optional[str] = None,
    sample_rate: float = 1.0
) -> Any:
    """初始化 OpenTelemetry 追踪
    
    Args:
        service_name: 服务名称
        service_version: 服务版本
        exporter_type: 导出器类型 (console/otlp/jaeger)
        otlp_endpoint: OTLP endpoint (如 http://localhost:4317)
        sample_rate: 采样率 (0.0-1.0)
    
    Returns:
        Tracer 实例
    """
    global _tracer, _provider

    if not _OPENTELEMETRY_AVAILABLE:
        logger.info("opentelemetry not available, returning no-op tracer")
        _tracer = _NoOpTracer()
        return _tracer

    if _tracer is not None:
        return _tracer

    # 创建 Resource
    resource = Resource.create({
        SERVICE_NAME: service_name,
        SERVICE_VERSION: service_version,
        "service.namespace": "packcv",
        "service.instance.id": os.getenv("HOSTNAME", "local"),
    })

    # 创建 TracerProvider
    _provider = TracerProvider(resource=resource)

    # 选择导出器
    if exporter_type == "console":
        exporter = ConsoleSpanExporter()
    elif exporter_type == "otlp" and otlp_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
        except ImportError:
            logger.warning("OTLP exporter not available, fallback to console")
            exporter = ConsoleSpanExporter()
    else:
        exporter = ConsoleSpanExporter()

    # 添加 Span Processor
    _provider.add_span_processor(BatchSpanProcessor(exporter))

    # 设置全局 TracerProvider
    trace.set_tracer_provider(_provider)

    # 获取 Tracer
    _tracer = trace.get_tracer(service_name, service_version)

    logger.info(f"OpenTelemetry initialized: {service_name}@{service_version}, exporter={exporter_type}")

    return _tracer


def get_tracer() -> Any:
    """获取全局 Tracer"""
    global _tracer
    if _tracer is None:
        _tracer = init_tracing()
    return _tracer


def instrument_fastapi(app: Any) -> None:
    """自动注入 FastAPI 追踪"""
    if not _OPENTELEMETRY_AVAILABLE or FastAPIInstrumentor is None:
        logger.info("opentelemetry instrumentation not available, skipping")
        return
    FastAPIInstrumentor.instrument_app(app)
    logger.info("FastAPI instrumented for OpenTelemetry")


def trace_llm_call(
    provider: str,
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    latency_ms: float = 0,
    success: bool = True,
    error_msg: Optional[str] = None,
    tenant_id: Optional[str] = None,
    scenario: Optional[str] = None
):
    """记录 LLM 调用追踪
    
    Args:
        provider: LLM provider ID
        model: 模型名称
        input_tokens: 输入 token 数
        output_tokens: 输出 token 数
        latency_ms: 响应延迟 (毫秒)
        success: 是否成功
        error_msg: 错误信息
        tenant_id: 租户 ID
        scenario: 业务场景
    """
    tracer = get_tracer()
    
    with tracer.start_as_current_span("llm.call") as span:
        span.set_attribute("llm.provider", provider)
        span.set_attribute("llm.model", model)
        span.set_attribute("llm.input_tokens", input_tokens)
        span.set_attribute("llm.output_tokens", output_tokens)
        span.set_attribute("llm.latency_ms", latency_ms)
        span.set_attribute("llm.success", success)
        
        if tenant_id:
            span.set_attribute("tenant.id", tenant_id)
        if scenario:
            span.set_attribute("business.scenario", scenario)
        
        if success:
            span.set_status(Status(StatusCode.OK))
        else:
            span.set_status(Status(StatusCode.ERROR, error_msg or "LLM call failed"))
            if error_msg:
                span.set_attribute("error.message", error_msg)


def trace_redis_op(
    operation: str,
    key: str,
    latency_ms: float = 0,
    success: bool = True,
    tenant_id: Optional[str] = None
):
    """记录 Redis 操作追踪"""
    tracer = get_tracer()
    
    with tracer.start_as_current_span(f"redis.{operation}") as span:
        span.set_attribute("redis.operation", operation)
        span.set_attribute("redis.key", key)
        span.set_attribute("redis.latency_ms", latency_ms)
        span.set_attribute("redis.success", success)
        
        if tenant_id:
            span.set_attribute("tenant.id", tenant_id)
        
        if success:
            span.set_status(Status(StatusCode.OK))
        else:
            span.set_status(Status(StatusCode.ERROR))


def trace_workflow_node(
    node_name: str,
    input_type: str = "",
    output_type: str = "",
    duration_ms: float = 0,
    success: bool = True,
    tenant_id: Optional[str] = None,
    run_id: Optional[str] = None
):
    """记录 LangGraph 工作流节点追踪"""
    tracer = get_tracer()
    
    with tracer.start_as_current_span(f"workflow.node.{node_name}") as span:
        span.set_attribute("workflow.node", node_name)
        span.set_attribute("workflow.input_type", input_type)
        span.set_attribute("workflow.output_type", output_type)
        span.set_attribute("workflow.duration_ms", duration_ms)
        span.set_attribute("workflow.success", success)
        
        if tenant_id:
            span.set_attribute("tenant.id", tenant_id)
        if run_id:
            span.set_attribute("workflow.run_id", run_id)
        
        if success:
            span.set_status(Status(StatusCode.OK))
        else:
            span.set_status(Status(StatusCode.ERROR))


def traced(name: str) -> Callable:
    """装饰器: 自动创建 Span"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_tracer()
            with tracer.start_as_current_span(name) as span:
                span.set_attribute("function.name", func.__name__)
                try:
                    result = func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.set_attribute("error.type", type(e).__name__)
                    raise
        return wrapper
    return decorator


def shutdown_tracing() -> None:
    """关闭追踪 (flush pending spans)"""
    global _provider
    if _provider and hasattr(_provider, "shutdown"):
        _provider.shutdown()
        logger.info("OpenTelemetry shutdown complete")


class _NoOpSpan:
    """无 OpenTelemetry 时的占位 Span"""

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_status(self, *args: Any, **kwargs: Any) -> None:
        pass

    def record_exception(self, exc: Any) -> None:
        pass

    def end(self) -> None:
        pass

    def __enter__(self) -> "_NoOpSpan":
        return self

    def __exit__(self, *args: Any) -> None:
        pass


class _NoOpTracer:
    """无 OpenTelemetry 时的占位 Tracer"""

    def start_as_current_span(self, name: str, **kwargs: Any) -> _NoOpSpan:
        return _NoOpSpan()

    def start_span(self, name: str, **kwargs: Any) -> _NoOpSpan:
        return _NoOpSpan()