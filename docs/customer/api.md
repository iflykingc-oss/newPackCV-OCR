# 🔌 API 参考

> 完整的 REST + GraphQL API 文档,OpenAPI 规范可在 `/openapi-spec` 访问。

## 基础信息

- **Base URL**: `https://api.vibecoding.dev`
- **认证方式**: `Authorization: Bearer <API_KEY>`
- **Content-Type**: `application/json` (除上传文件外)
- **限流**: 默认 100 req/s,可申请提升

## 错误码

| HTTP | 错误码 | 含义 | 处理建议 |
|------|--------|------|---------|
| 400 | `INVALID_PARAMS` | 参数错误 | 检查参数格式 |
| 401 | `INVALID_API_KEY` | API Key 无效 | 检查 key 是否正确 |
| 402 | `QUOTA_EXCEEDED` | 配额耗尽 | 升级套餐或等待下月 |
| 404 | `NOT_FOUND` | 资源不存在 | 检查 URL/任务 ID |
| 429 | `RATE_LIMITED` | 触发限流 | 等待 Retry-After 秒 |
| 500 | `INTERNAL_ERROR` | 服务异常 | 重试 (SDK 自动处理) |
| 503 | `SERVICE_UNAVAILABLE` | 服务暂不可用 | 重试或降级 |

## 核心端点

### POST /api/v1/extract

**同步提取** (单张图片/单个文档,30 秒内返回)

**请求**:
```json
{
  "image": "https://example.com/r.jpg",   // 图片 URL/二进制/base64
  "document": "https://example.com/c.pdf", // 文档 URL/二进制
  "scenario": "finance_receipt",            // 场景
  "locale": "zh-CN",                       // 错误消息语言
  "engine_tier": null,                     // 引擎优先级 (null=自动)
  "webhook_url": null                      // 异步回调 URL
}
```

**响应** (200):
```json
{
  "request_id": "req_abc123",
  "scenario": "finance_receipt",
  "fields": {
    "票据类型": "电子发票",
    "金额": "¥1,234.56",
    "日期": "2024-01-15",
    "商户": "ACME咖啡"
  },
  "confidence": 0.95,
  "engine_used": "paddleocr_vl",
  "latency_ms": 1230,
  "raw_text": "...",
  "tables": [...],
  "barcode_results": [...],
  "stamp_results": [...],
  "created_at": "2024-01-15T10:30:00Z"
}
```

### POST /api/v1/extract/async

**异步任务** (适合大文件/批量,立即返回 task_id)

**请求**: 同 `/extract`

**响应** (202):
```json
{
  "task_id": "task_abc123",
  "status_url": "/api/v1/tasks/task_abc123"
}
```

### GET /api/v1/tasks/{task_id}

**查询异步任务状态**

**响应**:
```json
{
  "task_id": "task_abc123",
  "state": "completed",          // pending|processing|completed|failed
  "result": {...},               // completed 时存在
  "error": null,                 // failed 时存在
  "created_at": "...",
  "completed_at": "..."
}
```

### POST /api/v1/extract/batch

**批量同步** (最多 100 个文件,自动并发)

**请求**:
```json
{
  "images": ["url1", "url2", "url3"],
  "scenario": "finance_receipt"
}
```

**响应**:
```json
{
  "total": 3,
  "succeeded": 3,
  "failed": 0,
  "results": [
    {"request_id": "req_1", "fields": {...}, "confidence": 0.95, ...},
    {"request_id": "req_2", "fields": {...}, "confidence": 0.92, ...},
    {"request_id": "req_3", "fields": {...}, "confidence": 0.89, ...}
  ]
}
```

## 业务端点

### GET /api/v1/scenarios

列出所有支持的场景 Schema。

### GET /api/v1/usage

**响应**:
```json
{
  "current_period": "2024-01",
  "used": 5432,
  "quota": 100000,
  "remaining": 94568,
  "reset_at": "2024-02-01T00:00:00Z"
}
```

### GET /api/v1/quota

查询当前套餐配额和限制。

### POST /api/v1/webhooks

注册 Webhook (异步任务完成时推送)。

## GraphQL

GraphQL 端点: `/graphql`

```graphql
query {
  extract(image: "https://...", scenario: "finance_receipt") {
    requestId
    fields
    confidence
    engineUsed
  }
}
```

## Webhook 推送

注册 Webhook 后,异步任务完成时会 POST 到您的 URL:

**Payload**:
```json
{
  "event": "task.completed",
  "task_id": "task_abc123",
  "tenant_id": "tnt_xxx",
  "result": {...},
  "timestamp": "2024-01-15T10:30:00Z"
}
```

**签名验证** (Header `X-PackCV-Signature`):
```
HMAC-SHA256(timestamp + "." + body, your_webhook_secret)
```

## SDK

我们提供官方 SDK:
- 🐍 [Python](https://github.com/iflykingc-oss/newPackCV-OCR/tree/main/sdk/python) - `pip install packcv-ocr`
- 🟨 JavaScript/TypeScript - `npm install @packcv/sdk`
- 🐹 Go - `go get github.com/vibecoding/packcv-go`

## Rate Limiting

- **默认**: 100 req/s, 1000 req/min
- **Pro**: 500 req/s, 5000 req/min
- **Enterprise**: 自定义

响应头:
- `X-RateLimit-Limit`: 总配额
- `X-RateLimit-Remaining`: 剩余配额
- `X-RateLimit-Reset`: 重置时间 (Unix timestamp)
- `Retry-After`: 触发限流时等待秒数

## OpenAPI 规范

完整 OpenAPI 3.0 规范可访问:
- `/openapi-spec` - JSON 格式
- `/docs` - Swagger UI

## 支持

- 📧 邮件: support@vibecoding.dev
- 💬 Slack: [vibecoding-community.slack.com](https://vibecoding-community.slack.com)
- 🐛 Issues: [GitHub Issues](https://github.com/iflykingc-oss/newPackCV-OCR/issues)
- 📖 状态页: [status.vibecoding.dev](https://status.vibecoding.dev)
