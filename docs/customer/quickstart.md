# 🚀 5分钟快速开始

> 从注册到第一次API调用,只需 5 分钟。

## 1. 注册账号

访问 [https://vibecoding.dev/signup](https://vibecoding.dev/signup) 填写邮箱,即可获得:
- **API Key** (立即可用)
- **租户 ID** (管理后台)
- **30天免费试用** (1,000 次配额)

## 2. 安装 SDK

```bash
pip install packcv-ocr
```

或使用 HTTP API (无需 SDK):

```bash
curl -X POST "https://api.vibecoding.dev/api/v1/extract" \
  -H "Authorization: Bearer pck_live_xxx" \
  -H "Content-Type: application/json" \
  -d '{
    "image": "https://example.com/receipt.jpg",
    "scenario": "finance_receipt",
    "locale": "zh-CN"
  }'
```

## 3. 第一次调用

### Python

```python
from packcv import PackCVClient

client = PackCVClient(api_key="pck_live_xxx")
result = client.extract(
    image="receipt.jpg",
    scenario="finance_receipt",
)
print(result.fields)
# {'金额': '¥1,234.56', '日期': '2024-01-15', '商户': 'ACME咖啡'}
```

### JavaScript / TypeScript

```javascript
import { PackCVClient } from '@packcv/sdk';

const client = new PackCVClient({ apiKey: 'pck_live_xxx' });
const result = await client.extract({
  image: 'receipt.jpg',
  scenario: 'finance_receipt',
});
console.log(result.fields);
```

### cURL

```bash
curl -X POST "https://api.vibecoding.dev/api/v1/extract" \
  -H "Authorization: Bearer pck_live_xxx" \
  -F "image=@receipt.jpg" \
  -F "scenario=finance_receipt"
```

## 4. 场景选择

| 您的文档 | 推荐场景 |
|---------|---------|
| 产品包装/标签 | `packaging` |
| 发票/收据/回单 | `finance_receipt` |
| 银行流水/对账单 | `finance_statement` |
| 药品包装/说明书 | `pharmaceutical` |
| 合同/协议 | `contract` |
| 身份证/护照 | `id_card` |
| 快递单/运单 | `logistics` |
| 不确定 | `auto` (自动识别) |

## 5. 错误处理

```python
from packcv import (
    PackCVClient,
    QuotaExceededError,
    RateLimitError,
)

client = PackCVClient(api_key="pck_live_xxx")

try:
    result = client.extract(image="r.jpg", scenario="finance_receipt")
except QuotaExceededError:
    print("配额用完,请升级到 Pro")
except RateLimitError as e:
    print(f"限流,{e.retry_after}s 后重试")
```

## 下一步

- 📖 [完整 API 参考](./api.md)
- 🎯 [8 大场景 Schema](./scenarios.md)
- 📨 [Webhook 集成](./webhooks.md)
- ⭐ [最佳实践](./best-practices.md)
- 💼 [客户案例](../case-studies/)
