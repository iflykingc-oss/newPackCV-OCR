"""FastAPI应用入口 - PackCV-OCR Phase 2 API服务

启动命令:
    uvicorn api.main:app --host 0.0.0.0 --port 9000 --workers 4

环境变量:
    REDIS_URL: Redis连接URL（默认redis://localhost:6379/0）
    ENV: 运行环境 (dev/staging/prod)
    LANGCHAIN_TRACING_V2: 启用LangSmith追踪
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.middleware.auth import AuthMiddleware
from api.middleware.metrics import MetricsMiddleware, metrics_endpoint
from api.routes import system, workflow, billing

# Phase 3 新增路由（可选加载）
_optional_routers = {}
for _rname in ("providers", "dashboard", "openapi", "webhooks", "health", "phase5", "phase6", "monitoring", "api_lifecycle", "phase7"):
    try:
        _mod = __import__(f"api.routes.{_rname}", fromlist=["router"])
        _optional_routers[_rname] = _mod.router
    except Exception as e:
        logging.getLogger("packcv-ocr.api").warning(f"路由 {_rname} 未加载: {e}")

# Web 管理后台
try:
    from web import router as web_router, mount_static
    _optional_routers["web"] = web_router
    logging.getLogger("packcv-ocr.api").info("✅ Web 管理后台路由已注册")
except Exception as e:
    logging.getLogger("packcv-ocr.api").warning(f"Web 路由未加载: {e}")


# ============= 日志 =============

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("packcv-ocr.api")


# ============= 生命周期 =============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动/关闭钩子"""
    # 启动
    logger.info("🚀 PackCV-OCR API 启动中...")
    env = os.getenv("ENV", "dev")
    logger.info(f"环境: {env}")

    # 初始化演示租户（仅开发环境）
    if env == "dev":
        try:
            from tenancy.api_key_manager import APIKeyManager

            APIKeyManager.setup_demo_tenants()
            logger.info("✅ 演示租户初始化完成")
        except Exception as e:
            logger.warning(f"演示租户初始化失败: {e}")

    # 初始化Redis连接检查
    try:
        from utils.redis_client import redis_client

        if redis_client.health_check():
            logger.info("✅ Redis连接正常")
        else:
            logger.warning("⚠️ Redis连接异常，部分限流功能将降级")
    except Exception as e:
        logger.warning(f"Redis健康检查失败: {e}")

    logger.info("✅ PackCV-OCR API 启动完成")
    yield
    # 关闭
    logger.info("PackCV-OCR API 关闭中...")


# ============= FastAPI应用 =============

app = FastAPI(
    title="PackCV-OCR API",
    description="多场景图片/文档智能信息提取API - 多租户SaaS版本",
    version="7.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "X-Request-Id",
        "X-Tenant-Id",
        "X-RateLimit-Limit",
        "X-RateLimit-Remaining",
        "X-RateLimit-Reset",
    ],
)

# 鉴权+限流中间件
app.add_middleware(MetricsMiddleware)
app.add_middleware(AuthMiddleware)

# Prometheus metrics端点（跳过鉴权）
app.add_route("/metrics", metrics_endpoint, methods=["GET"])

# 路由
app.include_router(system.router)
app.include_router(workflow.router)
app.include_router(billing.router)

# Phase 4 健康探针（断路器 + 三探针）
try:
    from api.routes import health as health_routes
    app.include_router(health_routes.router)
    logger.info("✅ 健康探针路由已注册")
except Exception as e:
    logger.error("健康探针路由加载失败: %s", e)

# Phase 5 智能增强 + 企业级 + 国际化
try:
    from api.routes import phase5
    app.include_router(phase5.router)
    logger.info("✅ Phase 5 路由已注册")
except Exception as e:
    logger.error("Phase 5 路由加载失败: %s", e)

# Phase 6 GraphQL 统一查询
try:
    from gql_api import graphql_app
    app.mount("/graphql", graphql_app)
    logger.info("✅ GraphQL 路由已挂载 (/graphql)")
except Exception as e:
    logger.warning(f"GraphQL 路由未加载: {e}")

# Phase 6 API 版本管理
try:
    from api.routes import api_lifecycle
    app.include_router(api_lifecycle.router)
    logger.info("✅ API 版本管理路由已注册")
except Exception as e:
    logger.warning(f"API 版本管理路由未加载: {e}")

# Phase 6 实时仪表盘
try:
    from api.routes import monitoring
    app.include_router(monitoring.router)
    logger.info("✅ 实时仪表盘路由已注册")
except Exception as e:
    logger.warning(f"实时仪表盘路由未加载: {e}")

# 注册 Phase 3 新路由
for _rname, _r in _optional_routers.items():
    app.include_router(_r)
    logger.info(f"✅ 路由已注册: {_rname}")

# Phase 7: 分布式追踪 + 配置热更新初始化
try:
    from tracing import init_tracing, instrument_fastapi
    init_tracing(service_name="packcv-api", service_version="7.0.0", exporter_type="console")
    instrument_fastapi(app)
    logger.info("✅ OpenTelemetry 追踪已启用")
except Exception as e:
    logger.warning(f"OpenTelemetry 追踪未启用: {e}")

try:
    from config_hotreload import init_configs
    init_configs()
    logger.info("✅ 配置热更新已启用")
except Exception as e:
    logger.warning(f"配置热更新未启用: {e}")

# 挂载 Web 静态资源
try:
    mount_static(app)
    logger.info("✅ Web 静态资源已挂载")
except Exception as e:
    logger.warning(f"Web 静态资源挂载失败: {e}")


# ============= 根路径 =============

@app.get("/")
async def root():
    """API根路径"""
    return {
        "service": "PackCV-OCR API",
        "version": "7.0.0",
        "description": "多场景图片/文档智能信息提取",
        "docs": "/docs",
        "endpoints": {
            "health": "GET /api/v1/health",
            "create_tenant": "POST /api/v1/admin/tenants",
            "demo_tenants": "POST /api/v1/admin/tenants/demo",
            "scenarios": "GET /api/v1/scenarios",
            "extract": "POST /api/v1/extract",
            "qa": "POST /api/v1/qa",
            "batch": "POST /api/v1/batch",
            "billing": {
                "record_usage": "POST /api/v1/billing/record",
                "get_usage": "GET /api/v1/billing/usage",
                "generate_invoice": "GET /api/v1/billing/invoice",
            },
            "audit": {
                "query_logs": "GET /api/v1/audit/logs",
            },
            "security": {
                "mask_text": "POST /api/v1/security/mask",
                "validate": "POST /api/v1/security/validate",
            },
            "degradation": {
                "policy": "GET /api/v1/degradation/policy",
            },
        },
    }


# ============= 全局异常 =============

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """全局异常处理"""
    import traceback

    logger.error(f"未处理异常: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": str(exc),
            "traceback": traceback.format_exc() if os.getenv("DEBUG") else None,
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=9000,
        reload=os.getenv("ENV") == "dev",
    )
