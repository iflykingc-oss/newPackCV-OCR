# PackCV-OCR Python SDK

[![PyPI](https://img.shields.io/pypi/v/packcv-ocr.svg)](https://pypi.org/project/packcv-ocr/)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

> 官方 Python SDK,3 行代码接入 VibeCoding-OCR 服务。

---

## 📦 安装

```bash
pip install packcv-ocr

# 异步支持 (可选)
pip install packcv-ocr[async]
```

## 🚀 快速开始

```python
from packcv import PackCVClient

# 1. 初始化客户端
client = PackCVClient(api_key="pck_live_xxx")

# 2. 提取单张图片
result = client.extract(
    image="receipt.jpg",
    scenario="finance_receipt",
    locale="zh-CN",
)
print(result.fields)
# {'金额': '¥1,234.56', '日期': '2024-01-15', '商户': 'ACME咖啡'}

# 3. 批量处理
results = client.batch_extract(
    images=["r1.jpg", "r2.jpg", "r3.jpg"],
    scenario="finance_receipt",
)
print(f"成功率: {results.success_rate:.0%} ({results.succeeded}/{results.total})")
```

## 🎯 支持的场景

| 场景 | scenario 值 | 适用文档 |
|------|------------|---------|
| 📦 包装 | `packaging` | 产品包装、标签 |
| 💰 金融票据 | `finance_receipt` | 发票、收据、回单 |
| 🏦 银行流水 | `finance_statement` | 银行对账单 |
| 💊 医药 | `pharmaceutical` | 药品包装、说明书 |
| 📋 合同 | `contract` | 劳动合同、商业合同 |
| 🪪 证件 | `id_card` | 身份证、护照 |
| 🚚 物流 | `logistics` | 快递单、运单 |
| 📄 通用文档 | `general_document` | 任意文档 |
| 🤖 自动识别 | `auto` | 由系统判断 |

## 📚 完整 API 文档

### `client.extract(...)`

```python
result = client.extract(
    image="receipt.jpg",                      # 图片路径/URL/二进制
    document="contract.pdf",                  # 文档路径(可选)
    scenario="finance_receipt",               # 场景 (默认 auto)
    locale="zh-CN",                           # 错误消息语言
    engine_tier=1,                            # 强制使用某引擎 (None=自动)
    webhook_url="https://...",                # 异步回调
)
```

返回 `ExtractResult` 对象,关键属性:
- `result.fields` - 提取的字段字典
- `result.confidence` - 置信度 (0-1)
- `result.engine_used` - 实际使用的引擎
- `result.latency_ms` - 耗时
- `result.tables` - 表格 (列表)
- `result.barcode_results` - 条码结果
- `result.stamp_results` - 印章结果

### `client.batch_extract(...)`

并发批量处理,自动处理部分失败,返回 `BatchResult`。

### `client.extract_async(...)`

异步任务,适用于大文件/批量场景:
```python
task = client.extract_async(
    image="doc.pdf",
    scenario="contract",
    webhook_url="https://yours.com/webhook",
)
# 方式1: 阻塞等待
result = task.poll(timeout=300)
# 方式2: 主动查询
status = task.status()
```

### 业务 API

```python
# 查看支持的场景
scenarios = client.list_scenarios()

# 查询本月用量
usage = client.get_usage()
print(f"已用: {usage['used']}/{usage['quota']}")

# 查询配额
quota = client.get_quota()
```

## ⚙️ 错误处理

```python
from packcv import (
    PackCVClient,
    AuthenticationError,
    QuotaExceededError,
    RateLimitError,
    ServerError,
)

try:
    result = client.extract(image="r.jpg", scenario="finance_receipt")
except AuthenticationError:
    print("API Key 无效,请检查")
except QuotaExceededError as e:
    print(f"配额已用完 ({e.used}/{e.quota}),请升级套餐")
except RateLimitError as e:
    print(f"触发限流,{e.retry_after}s 后重试")
except ServerError as e:
    print(f"服务异常 (可重试): {e.message}")
```

## 🌍 异步客户端

```python
import asyncio
from packcv import AsyncPackCVClient

async def main():
    client = AsyncPackCVClient(api_key="pck_live_xxx")
    result = await client.extract(
        image="receipt.jpg",
        scenario="finance_receipt",
    )
    print(result.fields)

asyncio.run(main())
```

需要 `pip install packcv-ocr[async]`。

## 🔧 高级配置

```python
client = PackCVClient(
    api_key="pck_live_xxx",
    base_url="https://api.vibecoding.dev",   # 自托管endpoint
    timeout=60,                                # 单次请求超时
    max_retries=5,                             # 失败重试次数
)
```

## 📄 License

Apache 2.0 © [VibeCoding Team](https://github.com/iflykingc-oss/newPackCV-OCR)
