# 支持的提取场景

VibeCoding-OCR 自动检测 8 类常见业务场景，每类都有独立的字段 Schema。

## 场景清单

| 场景代码 | 名称 | 必填字段数 | 典型应用 |
|---------|------|----------|---------|
| `packaging` | 产品包装 | 9 | 包装盒合规审查 |
| `finance_receipt` | 金融票据 | 7 | 发票/收据识别 |
| `finance_statement` | 银行流水 | 8 | 流水单结构化 |
| `pharmaceutical` | 医药包装 | 10 | 药品说明书 |
| `contract` | 合同 | 8 | 商业合同抽取 |
| `id_card` | 证件 | 7 | 身份证/护照 |
| `logistics` | 物流单 | 9 | 运单/快递单 |
| `general_document` | 通用文档 | 3+ | 兜底方案 |

## 场景自动检测

```
输入图片
  ↓
VL 多模态分类(A~H) + 关键词正则匹配
  ↓
融合决策 → 场景代码
  ↓
加载对应 Schema + LLM Config
  ↓
场景化提取
```

## 自定义场景

如需添加新场景，参考 [场景开发指南](../development/custom-scenario.md)。
