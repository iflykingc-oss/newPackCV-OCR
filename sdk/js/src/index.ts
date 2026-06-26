/**
 * PackCV-OCR JavaScript/TypeScript SDK
 * 多租户文档智能提取服务客户端
 *
 * @example
 * ```typescript
 * import { PackCVClient } from '@packcv-ocr/sdk';
 *
 * const client = new PackCVClient({
 *   api_key: 'pk_live_xxxxx',
 *   base_url: 'https://api.packcv-ocr.io',
 * });
 *
 * const result = await client.extract(
 *   'https://example.com/invoice.pdf',
 *   { scenario: 'invoice', user_question: '提取发票金额' }
 * );
 * console.log(result.structured_data);
 * ```
 *
 * @packageDocumentation
 */

// 类型
export type {
  Tier,
  Scenario,
  ScenarioId,
  ExtractRequest,
  ExtractResponse,
  ExtractOptions,
  BatchItem,
  BatchResult,
  QAResponse,
  UsageStats,
  UsageByDay,
  UsageByProvider,
  Tenant,
  TenantCreate,
  ApiKey,
  HealthStatus,
  Provider,
  ProviderHealth,
  WebhookSubscription,
  WebhookDispatchRequest,
  WebhookDispatchResult,
} from './types';

// 异常
export {
  PackCVError,
  AuthenticationError,
  RateLimitError,
  QuotaExceededError,
  ValidationError,
  ServerError,
  TimeoutError,
  NetworkError,
  errorFromStatus,
} from './exceptions';

// 客户端
export { PackCVClient, type PackCVClientOptions } from './client';
export { AsyncPackCVClient, type AsyncPackCVOptions } from './async-client';
export { BrowserPackCVClient, type BrowserPackCVOptions } from './browser';

// 工具
export { sleep, withTimeout, retry, type RetryOptions } from './utils';

// 版本
export const VERSION = '1.0.0';
export const SDK_NAME = '@packcv-ocr/sdk';
