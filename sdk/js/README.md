# @packcv-ocr/sdk

[![npm version](https://badge.fury.io/js/@packcv-ocr%2Fsdk.svg)](https://www.npmjs.com/package/@packcv-ocr/sdk)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.4-blue.svg)](https://www.typescriptlang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**PackCV-OCR** 是多租户文档智能提取服务，本 SDK 提供 Node.js / 浏览器 / TypeScript 全栈接入能力。

## ✨ 特性

- 🎯 **完整 TypeScript 支持** - 严格模式 + 完整类型定义
- 🌐 **三端统一** - Node.js（同步/异步） + 浏览器（Fetch API）
- 🔁 **自动重试** - 指数退避 + 抖动
- ⏱️ **超时控制** - 可配置请求超时
- 🛡️ **精细异常** - 9 种异常类型覆盖全部业务错误
- 📦 **轻量** - 零核心依赖（仅 node-fetch）

## 📦 安装

```bash
npm install @packcv-ocr/sdk
# 或
yarn add @packcv-ocr/sdk
# 或
pnpm add @packcv-ocr/sdk
```

## 🚀 快速开始

### Node.js（异步，推荐）

```typescript
import { PackCVClient } from '@packcv-ocr/sdk';

const client = new PackCVClient({
  api_key: process.env.PACKCV_API_KEY!,
  base_url: 'https://api.packcv-ocr.io',
  timeout: 60000,
  max_retries: 3,
});

// 1. 单文档提取
const result = await client.extract(
  'https://example.com/invoice.pdf',
  { scenario: 'invoice', user_question: '提取发票金额' }
);
console.log(result.structured_data);  // { amount: 1234.56, ... }
console.log(result.qa_answer);         // "发票金额为 1234.56 元"
console.log(result.confidence);        // 0.95

// 2. 批量提取（自动并发）
const batch = await client.batch_extract([
  { file_url: 'https://example.com/inv1.pdf', scenario: 'invoice' },
  { file_url: 'https://example.com/inv2.pdf', scenario: 'invoice' },
]);
console.log(`${batch.succeeded}/${batch.total} succeeded`);

// 3. 列出支持的场景
const scenarios = await client.list_scenarios();
for (const s of scenarios) {
  console.log(`${s.scenario_id}: ${s.name}`);
}
```

### 浏览器

```typescript
import { BrowserPackCVClient } from '@packcv-ocr/sdk';

const client = new BrowserPackCVClient({
  api_key: 'pk_live_xxxxx',
});

const result = await client.extract(fileInput.files[0] /* File */, {
  scenario: 'id_card',
});
```

### 异步并发客户端（高吞吐）

```typescript
import { AsyncPackCVClient } from '@packcv-ocr/sdk';

const client = new AsyncPackCVClient({
  api_key: process.env.PACKCV_API_KEY!,
  max_concurrent: 10,
});

// 1000 文档批量处理（自动分批并发 10）
const result = await client.batch_extract(urls.map(url => ({ file_url: url })));
```

## 📋 支持的 API

| 方法 | 说明 |
|------|------|
| `extract(file_url, options)` | 文档结构化提取 |
| `qa(file_url, question, options)` | 文档问答 |
| `batch_extract(items)` | 批量提取 |
| `list_scenarios()` | 列出支持的场景 |
| `list_tenants()` | 列出租户（管理员） |
| `create_tenant(data)` | 创建租户 |
| `get_tenant(id)` | 查询租户详情 |
| `delete_tenant(id)` | 删除租户 |
| `list_api_keys(tenant_id)` | 列出 API Key |
| `create_api_key(tenant_id, name)` | 创建 API Key |
| `revoke_api_key(tenant_id, key_id)` | 撤销 API Key |
| `get_usage(tenant_id, period)` | 查询用量统计 |
| `health()` | 服务健康检查 |
| `list_providers()` | 列出 LLM Provider |
| `get_provider_health(name)` | Provider 健康状态 |

## 🛡️ 异常处理

SDK 抛出 `PackCVError` 及其子类：

```typescript
import {
  PackCVError,
  AuthenticationError,
  RateLimitError,
  QuotaExceededError,
  ValidationError,
  ServerError,
  TimeoutError,
  NetworkError,
} from '@packcv-ocr/sdk';

try {
  await client.extract(url);
} catch (err) {
  if (err instanceof AuthenticationError) {
    console.error('API Key 无效');
  } else if (err instanceof RateLimitError) {
    console.error(`限流，${err.retry_after}秒后重试`);
  } else if (err instanceof QuotaExceededError) {
    console.error('本月配额已用完');
  } else if (err instanceof PackCVError) {
    console.error(`API 错误 [${err.code}]: ${err.message}`);
  }
}
```

## ⚙️ 配置

```typescript
new PackCVClient({
  api_key: 'pk_live_xxxxx',        // 必填
  base_url: 'https://api...',      // 可选，默认官方
  timeout: 60000,                  // 超时（毫秒）
  max_retries: 3,                  // 失败重试次数
  user_agent: 'my-app/1.0.0',      // 自定义 UA
});
```

## 🌍 环境

| 端 | 类 | 底层 |
|----|------|------|
| Node.js | `PackCVClient` | node-fetch |
| Node.js 高并发 | `AsyncPackCVClient` | node-fetch + 并发 |
| 浏览器 | `BrowserPackCVClient` | fetch API |

## 📝 License

MIT
