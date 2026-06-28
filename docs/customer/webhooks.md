# 📨 Webhook 集成

> 异步任务完成时,PackCV 会主动 POST 到您的 URL。

## 工作流

```
客户端                       PackCV                     您的服务器
  │                            │                            │
  │ POST /extract/async        │                            │
  ├───────────────────────────►│                            │
  │                            │                            │
  │ 202 { task_id: "..." }     │                            │
  │◄───────────────────────────┤                            │
  │                            │  (处理中...)               │
  │                            │                            │
  │                            │  POST /your-webhook        │
  │                            │  X-PackCV-Signature: ...   │
  │                            ├───────────────────────────►│
  │                            │                            │
  │                            │  200 OK                    │
  │                            │◄───────────────────────────┤
```

## 注册 Webhook

### 通过 API

```bash
curl -X POST "https://api.vibecoding.dev/api/v1/webhooks" \
  -H "Authorization: Bearer pck_live_xxx" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://my-server.com/webhook",
    "events": ["task.completed", "task.failed"],
    "description": "Production webhook"
  }'
```

**响应**:
```json
{
  "webhook_id": "wh_abc123",
  "url": "https://my-server.com/webhook",
  "secret": "whsec_xxx...",          // 用于签名验证
  "events": ["task.completed", "task.failed"],
  "created_at": "2024-01-15T10:30:00Z"
}
```

⚠️ **重要**: `secret` 仅在创建时返回一次,请妥善保管!

### 通过 SDK (Python)

```python
from packcv import PackCVClient

client = PackCVClient(api_key="pck_live_xxx")
webhook = client.webhooks.create(
    url="https://my-server.com/webhook",
    events=["task.completed", "task.failed"],
)
print(webhook.secret)  # 请立即保存!
```

---

## 接收 Webhook

### 请求格式

**Headers**:
```
Content-Type: application/json
User-Agent: PackCV-Webhook/1.0
X-PackCV-Signature: t=1705312200,v1=5257a869...
X-PackCV-Event: task.completed
X-PackCV-Delivery: dlv_abc123
X-PackCV-Task-Id: task_abc123
```

**Body**:
```json
{
  "event": "task.completed",
  "task_id": "task_abc123",
  "tenant_id": "tnt_xxx",
  "result": {
    "request_id": "req_abc123",
    "scenario": "finance_receipt",
    "fields": {
      "票据类型": "电子发票",
      "金额": "¥1,234.56"
    },
    "confidence": 0.95
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### 签名验证

**必须验证签名**,否则可能被恶意伪造请求。

```python
import hmac
import hashlib
import time

def verify_webhook(payload: bytes, signature: str, secret: str) -> bool:
    """
    signature 格式: "t=<timestamp>,v1=<hmac>"
    """
    parts = dict(p.split("=") for p in signature.split(","))
    timestamp = parts["t"]
    received_sig = parts["v1"]

    # 5 分钟时间窗口,防止重放攻击
    if abs(time.time() - int(timestamp)) > 300:
        return False

    # 计算 HMAC
    signed_payload = f"{timestamp}.".encode() + payload
    expected_sig = hmac.new(
        secret.encode(),
        signed_payload,
        hashlib.sha256,
    ).hexdigest()

    return hmac.compare_digest(received_sig, expected_sig)


# Flask 示例
from flask import Flask, request, abort

app = Flask(__name__)
WEBHOOK_SECRET = "whsec_xxx..."

@app.post("/webhook")
def webhook():
    signature = request.headers.get("X-PackCV-Signature", "")
    if not verify_webhook(request.data, signature, WEBHOOK_SECRET):
        abort(401)

    event = request.json
    if event["event"] == "task.completed":
        result = event["result"]
        # TODO: 处理结果
        save_to_database(result)
    elif event["event"] == "task.failed":
        error = event.get("error")
        # TODO: 处理失败

    return "", 200
```

### 幂等性

**Webhook 可能会重试多次** (网络异常时),必须保证业务幂等。

使用 `X-PackCV-Delivery` 作为唯一键:

```python
seen_deliveries = set()

@app.post("/webhook")
def webhook():
    delivery_id = request.headers["X-PackCV-Delivery"]
    if delivery_id in seen_deliveries:
        return "", 200  # 已处理过

    seen_deliveries.add(delivery_id)
    # TODO: 处理业务
    return "", 200
```

生产环境推荐使用 **Redis / 数据库** 存储 delivery_id,而不是内存。

---

## 事件类型

| 事件 | 触发时机 | Payload |
|------|---------|---------|
| `task.completed` | 任务成功 | `result` 字段 |
| `task.failed` | 任务失败 | `error` 字段 |
| `quota.warning` | 配额用到 80% | `usage` 字段 |
| `quota.exceeded` | 配额用尽 | `usage` 字段 |
| `tenant.suspended` | 租户被暂停 | 通知信息 |

---

## 重试策略

- **失败时自动重试**: 最多 5 次
- **重试间隔**: 1s, 5s, 30s, 2min, 10min (指数退避)
- **超时**: 单次 30 秒未响应视为失败
- **最终失败**: 记录到 `webhook_logs` 表,可手动重放

---

## 监控

### Webhook 状态

```bash
curl /api/v1/webhooks/wh_abc123 \
  -H "Authorization: Bearer ..."
```

**响应**:
```json
{
  "webhook_id": "wh_abc123",
  "url": "https://my-server.com/webhook",
  "status": "active",
  "stats": {
    "total_sent": 1234,
    "succeeded": 1220,
    "failed": 14,
    "success_rate": 0.988
  },
  "last_triggered_at": "2024-01-15T10:30:00Z"
}
```

### 失败告警

如果连续 10 次失败,PackCV 会:
1. 自动暂停该 Webhook
2. 发送邮件告警
3. 记录到审计日志

可手动恢复 (需修复服务端问题后):

```bash
curl -X POST /api/v1/webhooks/wh_abc123/reactivate \
  -H "Authorization: Bearer ..."
```

---

## 测试

### 测试模式

创建 Webhook 时加上 `?test=true` 会立即发送一个测试 payload:

```bash
curl -X POST "https://api.vibecoding.dev/api/v1/webhooks?test=true" \
  -d '{ ... }'
```

### 触发测试事件

```bash
curl -X POST "https://api.vibecoding.dev/api/v1/webhooks/wh_abc123/test" \
  -H "Authorization: Bearer ..."
```

---

## 最佳实践

1. **✅ HTTPS** - Webhook URL 必须使用 HTTPS
2. **✅ 快速响应** - 5 秒内返回 200,业务处理放入后台队列
3. **✅ 幂等性** - 使用 delivery_id 去重
4. **✅ 签名验证** - 必须验证
5. **✅ 监控** - 监控 success_rate
6. **❌ 不要在 Webhook 中调用 PackCV API** - 避免死循环
