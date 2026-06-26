# PackCV Python SDK

> Version: 1.0.0  
> Python: 3.10+

## 安装

```bash
pip install packcv-ocr
```

或从源码安装：
```bash
cd sdk/python && pip install -e .
```

## 快速开始

```python
from packcv import PackCVClient

with PackCVClient(
    api_key="pk_live_xxxxxxxx",
    base_url="https://api.packcv.example.com",
    timeout=30.0,
) as client:
    result = client.extract(
        file_url="https://example.com/license.jpg",
        scenario="business_license",
        user_question="提取法人和注册资本"
    )
    
    print(f"标题: {result.structured_data['title']}")
    print(f"置信度: {result.structured_data['confidence']}")
    print(f"问答: {result.qa_answer}")
    print(f"Token 用量: {result.usage.total_tokens}")
```

## 异步客户端

```python
import asyncio
from packcv import AsyncPackCVClient

async def main():
    async with AsyncPackCVClient(api_key="pk_live_xxx") as client:
        result = await client.extract(
            file_url="https://example.com/x.jpg",
            scenario="id_card",
        )
        print(result.qa_answer)

asyncio.run(main())
```

## 批量提取

```python
results = client.batch_extract(items=[
    {"file_url": "https://...", "scenario": "id_card"},
    {"file_url": "https://...", "scenario": "invoice"},
    {"file_url": "https://...", "scenario": "contract"},
])

for r in results:
    print(f"{r.run_id}: {r.status}")
```

## 健康检查

```python
status = client.health()
print(status)  # {'status': 'ok', ...}
```

## 错误处理

```python
from packcv import (
    PackCVError,
    PackCVAuthError,
    PackCVRateLimitError,
    PackCVQuotaError,
    PackCVValidationError,
    PackCVServerError,
)

try:
    result = client.extract(file_url="...", scenario="...")
except PackCVRateLimitError as e:
    print(f"限流，{e.retry_after} 秒后重试")
except PackCVQuotaError as e:
    print(f"配额用尽: {e}")
except PackCVAuthError:
    print("API Key 无效")
except PackCVServerError as e:
    print(f"服务错误: {e}")
except PackCVError as e:
    print(f"未知错误: {e}")
```

## 重试配置

```python
client = PackCVClient(
    api_key="pk_live_xxx",
    max_retries=5,             # 最多重试 5 次
    retry_backoff_factor=0.5,  # 退避因子
)
```

## 环境变量

- `PACKCV_API_KEY` - API Key
- `PACKCV_BASE_URL` - 服务地址（默认 `http://localhost:9001`）

## 完整 API

| 方法 | 说明 |
|------|------|
| `extract(file_url, scenario, user_question)` | 单次提取 |
| `qa(file_url, question)` | 问答 |
| `batch_extract(items)` | 批量提取 |
| `list_scenarios()` | 列出场景 |
| `health()` | 健康检查 |

## License

MIT
