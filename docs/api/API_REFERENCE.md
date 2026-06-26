# PackCV API Reference

> Version: 1.0.0  
> Base URL: `https://api.packcv.example.com`  
> Protocol: HTTP/1.1, JSON, UTF-8

## 通用约定

### 鉴权

所有受保护端点需要 Bearer Token：

```http
Authorization: Bearer pk_live_xxxxxxxx
```

### 多租户

`X-Tenant-Id` 头标识租户上下文：

```http
X-Tenant-Id: tenant_demo
```

### 限流响应

超出限流时返回 `429 Too Many Requests`：

```json
{
  "detail": "Rate limit exceeded",
  "retry_after": 30,
  "limit": "100/minute"
}
```

### 错误格式

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "file_url 不能为空",
    "request_id": "req_abc123"
  }
}
```

---

## 核心工作流

### 1. 结构化提取

```http
POST /api/v1/extract
```

**请求**:
```json
{
  "input_file": {
    "url": "https://example.com/license.jpg",
    "file_type": "image"
  },
  "scenario": "business_license",
  "user_question": "提取法人和注册资本"
}
```

**响应 200**:
```json
{
  "run_id": "uuid",
  "scenario": "business_license",
  "structured_data": {
    "title": "...",
    "document_type": "营业执照",
    "confidence": 0.95,
    "key_points": "...",
    "main_content": "..."
  },
  "qa_answer": "...",
  "usage": {
    "input_tokens": 1234,
    "output_tokens": 567,
    "total_tokens": 1801,
    "model": "doubao-pro"
  }
}
```

**支持场景**:
- `id_card` - 身份证
- `business_license` - 营业执照
- `contract` - 合同
- `invoice` - 发票
- `bank_card` - 银行卡
- `resume` - 简历
- `auto` - 自动识别

### 2. 问答

```http
POST /api/v1/qa
```

**请求**:
```json
{
  "input_file": {"url": "https://...", "file_type": "image"},
  "user_question": "法人是谁？"
}
```

### 3. 批量提取

```http
POST /api/v1/batch
```

**请求**:
```json
{
  "items": [
    {"file_url": "...", "scenario": "id_card"},
    {"file_url": "...", "scenario": "invoice"}
  ]
}
```

**限制**: 单批最多 100 项。

---

## 租户管理

### 创建租户
```http
POST /api/v1/admin/tenants
```

### 获取租户
```http
GET /api/v1/admin/tenants/{tenant_id}
```

### 列出所有租户
```http
GET /api/v1/admin/tenants
```

---

## 计费

### 查询用量
```http
GET /api/v1/billing/usage/{tenant_id}?month=2024-01
```

### 生成账单
```http
POST /api/v1/billing/invoice
```

---

## WebHook

### 订阅
```http
POST /webhooks/subscribe
```

**请求**:
```json
{
  "tenant_id": "demo",
  "url": "https://your-server.com/webhook",
  "events": ["task.completed", "billing.usage"],
  "secret": "your-hmac-secret-min-8-chars"
}
```

### 同步分发
```http
POST /webhooks/dispatch
```

### 异步分发
```http
POST /webhooks/dispatch-async
```

### 投递统计
```http
GET /webhooks/stats/{tenant_id}
```

### 死信队列
```http
GET /webhooks/dlq
POST /webhooks/dlq/{record_id}/replay
```

### 事件类型

| 类型 | 描述 |
|------|------|
| `task.created` | 任务创建 |
| `task.started` | 任务开始 |
| `task.completed` | 任务完成 |
| `task.failed` | 任务失败 |
| `billing.usage` | 用量上报 |
| `billing.quota_exceeded` | 配额超限 |
| `billing.invoice_ready` | 账单生成 |
| `tenant.created` | 新租户 |
| `tenant.suspended` | 租户暂停 |
| `apikey.rotated` | Key 轮换 |
| `system.degraded` | 系统降级 |
| `system.recovered` | 系统恢复 |

### 签名验证

每个 Webhook 请求包含 `X-PackCV-Signature` 头：

```
X-PackCV-Signature: t=1234567890,v1=abc123...
```

验证（Python 示例）：
```python
import hmac, hashlib

def verify(payload: bytes, header: str, secret: str) -> bool:
    parts = dict(p.split('=', 1) for p in header.split(','))
    ts, sig = parts['t'], parts['v1']
    signed = f"{ts}.".encode() + payload
    expected = hmac.new(secret.encode(), signed, hashlib.sha256).hexdigest()
    return hmac.compare_digest(sig, expected)
```

### 重试策略

- 最多 6 次尝试（1 次 + 5 次重试）
- 延迟序列: 1s, 5s, 25s, 125s, 625s
- 全部失败 → DLQ，可手动 replay

---

## LLM Provider

### 列出 Provider
```http
GET /providers
```

### 列出模型
```http
GET /providers/models
```

### 路由策略
```http
GET /providers/routing
```

### 健康状态
```http
GET /providers/health
```

---

## 监控

### Prometheus 指标
```http
GET /metrics
```

24 个自定义指标：
- `packcv_http_requests_total`
- `packcv_workflow_duration_seconds`
- `packcv_llm_tokens_total`
- `packcv_tenant_quota_usage`
- ...

---

## OpenAPI 规范

```http
GET /openapi-spec
```

返回完整 OpenAPI 3.0 规范，可用于 SDK 生成、API 客户端配置。

---

## 管理后台

- `/` - Dashboard 概览
- `/tenants` - 租户管理
- `/usage` - 用量分析
- `/billing` - 账单
- `/settings` - API Key
