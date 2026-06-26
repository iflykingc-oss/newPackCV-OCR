# PackCV 集成指南

本文档提供 4 种典型场景的端到端集成示例。

## 场景 1: Web 后台表单

使用 [Python SDK](../sdk/python/README.md) 处理用户上传的身份证：

```python
from packcv import PackCVClient, PackCVError

client = PackCVClient(
    api_key="pk_live_xxx",
    base_url="https://api.packcv.example.com"
)

try:
    result = client.extract(
        file_url="https://your-cdn.com/uploads/idcard.jpg",
        scenario="id_card",
        user_question="提取身份证号和姓名"
    )
    print(f"姓名: {result.structured_data['name']}")
    print(f"身份证号: {result.structured_data['id_number']}")
    print(f"置信度: {result.structured_data['confidence']}")
except PackCVError as e:
    if e.status_code == 429:
        print(f"限流，请 {e.retry_after} 秒后重试")
    else:
        print(f"错误: {e}")
finally:
    client.close()
```

## 场景 2: Node.js 后端服务

```typescript
import { PackCVClient } from '@packcv/sdk';

const client = new PackCVClient({
  apiKey: process.env.PACKCV_API_KEY!,
  baseUrl: 'https://api.packcv.example.com',
});

const result = await client.extract({
  fileUrl: 'https://your-bucket.s3.amazonaws.com/contract.pdf',
  scenario: 'contract',
  userQuestion: '提取甲乙双方、金额、有效期',
});

console.log('合同主体:', result.structured_data.parties);
console.log('金额:', result.structured_data.amount);
```

## 场景 3: 浏览器端

```typescript
import { PackCVBrowserClient } from '@packcv/sdk/browser';

const client = new PackCVBrowserClient({
  apiKey: 'pk_public_xxx',  // 公开 API Key
  baseUrl: 'https://api.packcv.example.com',
});

const fileInput = document.querySelector('#file') as HTMLInputElement;
const file = fileInput.files![0];

const result = await client.extract({
  file,  // 直接传 File 对象
  scenario: 'auto',
  userQuestion: '提取所有信息',
});

console.log(result.qa_answer);
```

## 场景 4: AI Agent (Claude Desktop / Cursor)

PackCV 提供 **MCP Server**，让 Claude / Cursor 直接调用：

配置 Claude Desktop (`claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "packcv": {
      "command": "python",
      "args": ["-m", "mcp_server"],
      "env": {
        "PACKCV_BASE_URL": "https://api.packcv.example.com",
        "PACKCV_API_KEY": "pk_live_xxx"
      }
    }
  }
}
```

配置 Cursor (`~/.cursor/mcp.json`):
```json
{
  "mcpServers": {
    "packcv": {
      "url": "http://localhost:9002/sse",
      "transport": "sse"
    }
  }
}
```

之后 Claude 可以直接使用：
```
请使用 packcv 的 extract_document 工具提取这张营业执照的法人信息
```

## 场景 5: 异步事件订阅（WebHook）

```python
# 1. 订阅事件
import httpx
r = httpx.post(
    "https://api.packcv.example.com/webhooks/subscribe",
    headers={"Authorization": "Bearer pk_live_xxx"},
    json={
        "tenant_id": "demo",
        "url": "https://your-server.com/webhook-receiver",
        "events": ["task.completed", "billing.quota_exceeded"],
        "secret": "your-shared-secret-12345"
    }
)
subscription_id = r.json()["subscription_id"]
print(f"订阅成功: {subscription_id}")

# 2. 接收端验证签名
from fastapi import FastAPI, Request, HTTPException
import hmac, hashlib

app = FastAPI()

@app.post("/webhook-receiver")
async def receive(request: Request):
    body = await request.body()
    sig_header = request.headers.get("X-PackCV-Signature", "")
    
    parts = dict(p.split("=", 1) for p in sig_header.split(",") if "=" in p)
    ts = parts.get("t")
    sig = parts.get("v1")
    signed = f"{ts}.".encode() + body
    expected = hmac.new(
        b"your-shared-secret-12345", signed, hashlib.sha256
    ).hexdigest()
    
    if not hmac.compare_digest(sig, expected):
        raise HTTPException(401, "Invalid signature")
    
    event = await request.json()
    if event["event_type"] == "task.completed":
        # 处理任务完成
        ...
    
    return {"status": "ok"}
```

## 错误处理最佳实践

### 指数退避
```python
import time

def call_with_retry(fn, max_attempts=3):
    for attempt in range(max_attempts):
        try:
            return fn()
        except PackCVRateLimitError as e:
            if attempt == max_attempts - 1:
                raise
            time.sleep(min(2 ** attempt, 30))
        except PackCVError:
            raise
```

### 幂等性
所有写操作支持 `Idempotency-Key` 头：
```python
result = client.extract(
    file_url="...",
    scenario="id_card",
    idempotency_key="order-12345-id-card",
)
```

## 性能建议

| 场景 | 并发 | 建议 |
|------|------|------|
| 单次提取 | - | 直接调用 |
| 批量 100 项 | 10 | 拆分 10 批并发 |
| 高频 1000+QPS | - | 使用 WebHook 异步 |

## SLA

- 可用性: **99.9%**
- P95 延迟: < 3s（标准场景）
- P99 延迟: < 10s
- 多区域容灾
