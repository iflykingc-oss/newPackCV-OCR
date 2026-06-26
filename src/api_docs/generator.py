#!/usr/bin/env python3
"""API 文档增强模块
功能:
- 自动生成 OpenAPI 示例
- 错误码表文档
- SDK 使用指南端点
- API 变更历史
"""
import os
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# 错误码定义
ERROR_CODES: Dict[int, Dict[str, Any]] = {
    # 通用错误 100xxx
    100001: {"code": "INVALID_REQUEST", "zh": "请求参数格式错误", "en": "Invalid request format", "ja": "リクエスト形式エラー"},
    100002: {"code": "MISSING_PARAMETER", "zh": "缺少必填参数", "en": "Missing required parameter", "ja": "必須パラメータ不足"},
    100003: {"code": "INVALID_FILE_TYPE", "zh": "不支持该文件类型", "en": "Unsupported file type", "ja": "サポートされていないファイルタイプ"},
    
    # 认证错误 101xxx
    101001: {"code": "UNAUTHORIZED", "zh": "未提供有效的API密钥", "en": "No valid API key provided", "ja": "有効なAPIキーが提供されていません"},
    101002: {"code": "API_KEY_EXPIRED", "zh": "API密钥已过期", "en": "API key expired", "ja": "APIキーが期限切れ"},
    101003: {"code": "TENANT_DISABLED", "zh": "租户已被禁用", "en": "Tenant disabled", "ja": "テナントが無効化されています"},
    
    # 配额错误 102xxx  
    102001: {"code": "RATE_LIMIT_EXCEEDED", "zh": "请求频率超过限制", "en": "Rate limit exceeded", "ja": "レート制限超過"},
    102002: {"code": "MONTHLY_QUOTA_EXCEEDED", "zh": "月度配额已耗尽", "en": "Monthly quota exhausted", "ja": "月次クォータ枯渇"},
    102003: {"code": "CONCURRENT_LIMIT", "zh": "并发请求数超限", "en": "Concurrent request limit", "ja": "同時接続数制限"},
    
    # 业务错误 103xxx
    103001: {"code": "OCR_FAILED", "zh": "OCR识别失败", "en": "OCR recognition failed", "ja": "OCR認識失敗"},
    103002: {"code": "LLM_CALL_FAILED", "zh": "LLM调用失败", "en": "LLM call failed", "ja": "LLM呼び出し失敗"},
    103003: {"code": "SCENARIO_NOT_SUPPORTED", "zh": "不支持该业务场景", "en": "Scenario not supported", "ja": "シナリオ未サポート"},
    
    # 系统错误 104xxx
    104001: {"code": "INTERNAL_ERROR", "zh": "系统内部错误", "en": "Internal system error", "ja": "内部システムエラー"},
    104002: {"code": "SERVICE_UNAVAILABLE", "zh": "服务暂时不可用", "en": "Service temporarily unavailable", "ja": "サービス一時利用不可"},
    104003: {"code": "CIRCUIT_BREAKER_OPEN", "zh": "熔断器已打开", "en": "Circuit breaker open", "ja": "回路遮断器開放"},
}


class APIExample(BaseModel):
    """API 示例"""
    endpoint: str = Field(..., description="API路径")
    method: str = Field(default="POST", description="HTTP方法")
    request_example: Dict[str, Any] = Field(default_factory=dict, description="请求示例")
    response_example: Dict[str, Any] = Field(default_factory=dict, description="响应示例")
    notes: str = Field(default="", description="使用说明")


class APIChange(BaseModel):
    """API 变更记录"""
    version: str = Field(..., description="版本号")
    date: str = Field(..., description="变更日期")
    changes: List[str] = Field(default_factory=list, description="变更列表")
    deprecated: List[str] = Field(default_factory=list, description="废弃端点")


# API 示例库
API_EXAMPLES: Dict[str, APIExample] = {
    "extract": APIExample(
        endpoint="/api/v1/extract",
        method="POST",
        request_example={
            "input_file": {"url": "https://example.com/license.jpg", "file_type": "image"},
            "scenarios": ["license"],
            "options": {"return_confidence": True}
        },
        response_example={
            "structured_data": {"legal_rep": "张三", "company_name": "XX科技有限公司"},
            "confidence": 0.95,
            "run_id": "abc123"
        },
        notes="支持图片/文档，自动识别场景并提取结构化数据"
    ),
    "qa": APIExample(
        endpoint="/api/v1/qa",
        method="POST",
        request_example={
            "input_file": {"url": "https://example.com/invoice.pdf", "file_type": "document"},
            "question": "发票金额是多少？",
            "options": {"return_source": True}
        },
        response_example={
            "answer": "发票金额为 ¥12,350.00",
            "source_text": "金额：12350.00元",
            "confidence": 0.92
        },
        notes="基于文档内容回答问题，支持溯源"
    ),
    "batch": APIExample(
        endpoint="/api/v1/batch",
        method="POST",
        request_example={
            "files": [
                {"url": "https://example.com/doc1.pdf"},
                {"url": "https://example.com/doc2.pdf"}
            ],
            "scenarios": ["invoice", "contract"],
            "callback_url": "https://your-server/callback"
        },
        response_example={
            "batch_id": "batch_xyz",
            "status": "processing",
            "estimated_time": 120
        },
        notes="批量处理，支持异步回调通知"
    ),
}

# API 变更历史
API_HISTORY: List[APIChange] = [
    APIChange(version="7.0.0", date="2026-06-25", changes=[
        "新增 GraphQL API 端点 (/graphql/)",
        "新增 分布式追踪 (OpenTelemetry)",
        "新增 配置热更新功能",
        "新增 数据血缘追踪",
    ]),
    APIChange(version="6.0.0", date="2026-06-20", changes=[
        "新增 API 版本管理模块",
        "新增 限流可视化仪表盘",
        "新增 Web 后台 i18n (中/英/日)",
    ]),
    APIChange(version="5.0.0", date="2026-06-15", changes=[
        "新增 LLM 响应缓存",
        "新增 A/B 测试框架",
        "新增 RBAC 权限管理",
        "新增 OIDC SSO 支持",
    ]),
    APIChange(version="4.0.0", date="2026-06-10", changes=[
        "新增 断路器熔断机制",
        "新增 优雅关停管理",
        "新增 CLI 工具 (packcv-cli)",
    ]),
]


def get_error_codes(locale: str = "zh") -> Dict[str, Any]:
    """获取错误码表
    
    Args:
        locale: 语言 (zh/en/ja)
    
    Returns:
        错误码字典
    """
    result = {}
    for code_int, info in ERROR_CODES.items():
        msg = info.get(locale, info.get("zh", ""))
        result[str(code_int)] = {
            "code": info["code"],
            "message": msg,
            "http_status": _get_http_status(code_int)
        }
    return {"total": len(result), "locale": locale, "errors": result}


def _get_http_status(error_code: int) -> int:
    """根据错误码推断 HTTP 状态码"""
    if 100000 <= error_code < 101000:
        return 400  # Bad Request
    elif 101000 <= error_code < 102000:
        return 401  # Unauthorized
    elif 102000 <= error_code < 103000:
        return 429  # Too Many Requests
    elif 103000 <= error_code < 104000:
        return 422  # Unprocessable Entity
    else:
        return 500  # Internal Server Error


def get_api_examples() -> Dict[str, Any]:
    """获取所有 API 示例"""
    return {
        "total": len(API_EXAMPLES),
        "examples": {k: v.model_dump() for k, v in API_EXAMPLES.items()}
    }


def get_api_history() -> Dict[str, Any]:
    """获取 API 变更历史"""
    return {
        "total": len(API_HISTORY),
        "history": [h.model_dump() for h in API_HISTORY]
    }


def get_sdk_guide() -> Dict[str, Any]:
    """获取 SDK 使用指南"""
    return {
        "python": {
            "install": "pip install packcv-sdk",
            "example": '''
from packcv import PackCVClient

client = PackCVClient(api_key="your-api-key")

# 单文件提取
result = client.extract(
    file_url="https://example.com/license.jpg",
    scenarios=["license"]
)
print(result.structured_data)

# 批量处理
batch = client.batch_extract(
    files=["url1", "url2"],
    scenarios=["invoice"]
)
''',
        },
        "javascript": {
            "install": "npm install packcv-sdk",
            "example": '''
import { PackCVClient } from 'packcv-sdk';

const client = new PackCVClient({ apiKey: 'your-api-key' });

// 单文件提取
const result = await client.extract({
  fileUrl: 'https://example.com/license.jpg',
  scenarios: ['license']
});
console.log(result.structuredData);

// 批量处理
const batch = await client.batchExtract({
  files: ['url1', 'url2'],
  scenarios: ['invoice']
});
''',
        },
        "curl": {
            "example": '''
# 提取接口
curl -X POST https://api.packcv.com/api/v1/extract \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"input_file":{"url":"https://example.com/doc.pdf"},"scenarios":["invoice"]}'

# GraphQL 查询
curl -X POST https://api.packcv.com/graphql/ \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"query":"{ health { live ready } cacheStats { hitRate } }"}'
''',
        }
    }