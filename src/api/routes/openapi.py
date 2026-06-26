"""
OpenAPI 3.0 规范生成 + 多语言SDK元信息
"""
import os
from typing import Any, Dict

from fastapi import APIRouter
from fastapi.openapi.utils import get_openapi

router = APIRouter(prefix="/openapi-spec", tags=["openapi"])


@router.get("", include_in_schema=False)
async def custom_openapi():
    """生成自定义OpenAPI规范"""
    from api.main import app

    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="PackCV-OCR API",
        version="7.0.0",
        description="""
# PackCV-OCR 多租户文档智能提取API

## 核心能力
- 📄 多场景文档识别（12种业务场景）
- 🌏 多语言支持（中/英/泰/越/印尼/马来）
- 🤖 VLM-First + 5级Fallback降级链
- 💰 4种计费模式（by_token/by_call/package/hybrid）
- 🛡️ 数据脱敏 + 全链路审计
- 📊 24个Prometheus业务指标

## 认证方式
所有API需要在Header中传递API Key：
```
X-API-Key: pk_test_xxx
X-API-Secret: sk_xxx  (可选，部分端点需要)
```

## 限流策略
- FREE: 10 RPM
- BASIC: 60+20 RPM
- PRO: 300+100 RPM
- ENTERPRISE: 1000+300 RPM
- FLAGSHIP: 5000+1500 RPM

## SLA承诺
- 可用性: 99.9%
- P99延迟: < 5s
- 错误率: < 0.1%
        """,
        routes=app.routes,
    )

    # 添加扩展字段
    openapi_schema["x-sdk-packages"] = {
        "python": {
            "package_name": "packcv-ocr",
            "install": "pip install packcv-ocr",
            "version": "1.0.0",
            "github": "https://github.com/packcv-ocr/packcv-ocr-python",
        },
        "javascript": {
            "package_name": "@packcv-ocr/sdk",
            "install": "npm install @packcv-ocr/sdk",
            "version": "1.0.0",
            "github": "https://github.com/packcv-ocr/packcv-ocr-js",
        },
        "go": {
            "package_name": "github.com/packcv-ocr/packcv-ocr-go",
            "install": "go get github.com/packcv-ocr/packcv-ocr-go",
            "version": "1.0.0",
        },
        "java": {
            "artifact_id": "com.packcv-ocr:sdk",
            "install": "implementation 'com.packcv-ocr:sdk:1.0.0'",
            "version": "1.0.0",
        },
    }

    openapi_schema["x-rate-limit"] = {
        "default_window": "1 minute",
        "tiers": {
            "free": "10 RPM",
            "basic": "60+20 RPM",
            "pro": "300+100 RPM",
            "enterprise": "1000+300 RPM",
            "flagship": "5000+1500 RPM",
        },
    }

    openapi_schema["x-billing-modes"] = ["by_token", "by_call", "package", "hybrid"]
    openapi_schema["x-sla"] = {
        "availability": "99.9%",
        "p99_latency_ms": 5000,
        "error_rate_max": 0.001,
    }

    app.openapi_schema = openapi_schema
    return openapi_schema
