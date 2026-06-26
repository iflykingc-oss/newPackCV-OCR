/**
 * PackCV 异步客户端（Node.js + 并发场景）
 * @module async-client
 */

import { errorFromStatus, PackCVError, NetworkError } from './exceptions';
import type {
  ExtractResponse,
  ExtractOptions,
  QAResponse,
  BatchItem,
  BatchResult,
  UsageStats,
  Tenant,
  TenantCreate,
  ApiKey,
  HealthStatus,
  Provider,
  Scenario,
  WebhookSubscription,
  WebhookDispatchRequest,
  WebhookDispatchResult,
} from './types';

export interface AsyncPackCVOptions {
  api_key: string;
  base_url?: string;
  timeout?: number;
  max_retries?: number;
  max_concurrent?: number;
}

const DEFAULT_BASE_URL = 'https://api.packcv-ocr.io';
const DEFAULT_TIMEOUT = 60000;

/** 异步 PackCV 客户端 */
export class AsyncPackCVClient {
  public readonly api_key: string;
  public readonly base_url: string;
  public readonly timeout: number;
  public readonly max_retries: number;
  public readonly max_concurrent: number;

  constructor(options: AsyncPackCVOptions) {
    this.api_key = options.api_key;
    this.base_url = (options.base_url ?? DEFAULT_BASE_URL).replace(/\/$/, '');
    this.timeout = options.timeout ?? DEFAULT_TIMEOUT;
    this.max_retries = options.max_retries ?? 3;
    this.max_concurrent = options.max_concurrent ?? 5;
  }

  private async _request<T>(method: string, path: string, body?: unknown): Promise<T> {
    const url = `${this.base_url}${path}`;
    // eslint-disable-next-line @typescript-eslint/no-implied-eval
    const nodeFetch: (url: string, init: { method: string; headers: Record<string, string>; body: string | null; signal: AbortSignal }) => Promise<{ status: number; json: () => Promise<unknown> }> = new Function('return require("node-fetch")')();
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeout);

    let res;
    try {
      res = await nodeFetch(url, {
        method,
        headers: {
          'Authorization': `Bearer ${this.api_key}`,
          'Content-Type': 'application/json',
        },
        body: body !== undefined ? JSON.stringify(body) : null,
        signal: controller.signal,
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      if (msg.includes('abort')) {
        throw new PackCVError(`Request timeout after ${this.timeout}ms`, 408, 'timeout');
      }
      throw new NetworkError(msg);
    } finally {
      clearTimeout(timer);
    }

    const status = res.status;
    let data: unknown;
    try {
      data = await res.json();
    } catch {
      data = {};
    }

    if (status >= 200 && status < 300) {
      return data as T;
    }
    const dataObj = (data ?? {}) as { message?: string; error?: string };
    const errMsg = dataObj.message ?? dataObj.error ?? `HTTP ${status}`;
    throw errorFromStatus(status, errMsg, data as Record<string, unknown>);
  }

  async extract(file_url: string, options: ExtractOptions = {}): Promise<ExtractResponse> {
    return this._request<ExtractResponse>('POST', '/api/v1/extract', { file_url, ...options });
  }

  async qa(file_url: string, question: string, options: Omit<ExtractOptions, 'user_question'> = {}): Promise<QAResponse> {
    return this._request<QAResponse>('POST', '/api/v1/qa', { file_url, user_question: question, ...options });
  }

  /** 并发批量提取 */
  async batch_extract(items: BatchItem[], concurrency: number = this.max_concurrent): Promise<BatchResult> {
    const results: BatchResult['items'] = [];
    let succeeded = 0;
    let failed = 0;

    // 简单实现：分批
    for (let i = 0; i < items.length; i += concurrency) {
      const batch = items.slice(i, i + concurrency);
      const promises = batch.map(async (item) => {
        try {
          const data = await this.extract(item.file_url, { scenario: item.scenario, user_question: item.user_question });
          return { file_url: item.file_url, status: 'success' as const, data: data.structured_data };
        } catch (err) {
          return { file_url: item.file_url, status: 'failed' as const, error: err instanceof Error ? err.message : String(err) };
        }
      });
      const batchResults = await Promise.all(promises);
      for (const r of batchResults) {
        results.push(r);
        if (r.status === 'success') succeeded++;
        else failed++;
      }
    }

    return { total: items.length, succeeded, failed, items: results };
  }

  async list_scenarios(): Promise<Scenario[]> {
    const data = await this._request<Scenario[] | { scenarios?: Scenario[] }>('GET', '/api/v1/scenarios');
    if (Array.isArray(data)) return data;
    return (data?.scenarios ?? []) as Scenario[];
  }

  async list_tenants(): Promise<Tenant[]> {
    const data = await this._request<Tenant[] | { tenants?: Tenant[] }>('GET', '/api/v1/admin/tenants');
    if (Array.isArray(data)) return data;
    return (data?.tenants ?? []) as Tenant[];
  }

  async create_tenant(data: TenantCreate): Promise<Tenant> {
    return this._request<Tenant>('POST', '/api/v1/admin/tenants', data);
  }

  async get_tenant(tenant_id: string): Promise<Tenant> {
    return this._request<Tenant>('GET', `/api/v1/admin/tenants/${tenant_id}`);
  }

  async delete_tenant(tenant_id: string): Promise<void> {
    await this._request('DELETE', `/api/v1/admin/tenants/${tenant_id}`);
  }

  async list_api_keys(tenant_id: string): Promise<ApiKey[]> {
    const data = await this._request<ApiKey[] | { keys?: ApiKey[] }>('GET', `/api/v1/admin/tenants/${tenant_id}/api-keys`);
    if (Array.isArray(data)) return data;
    return (data?.keys ?? []) as ApiKey[];
  }

  async create_api_key(tenant_id: string, name: string, scopes: string[] = ['read', 'write']): Promise<ApiKey> {
    return this._request<ApiKey>('POST', `/api/v1/admin/tenants/${tenant_id}/api-keys`, { name, scopes });
  }

  async get_usage(tenant_id: string, period?: string): Promise<UsageStats> {
    const q = period ? `?period=${period}` : '';
    return this._request<UsageStats>('GET', `/api/v1/billing/usage/${tenant_id}${q}`);
  }

  async health(): Promise<HealthStatus> {
    return this._request<HealthStatus>('GET', '/api/v1/system/health');
  }

  async list_providers(): Promise<Provider[]> {
    const data = await this._request<Provider[] | { providers?: Provider[] }>('GET', '/providers');
    if (Array.isArray(data)) return data;
    return (data?.providers ?? []) as Provider[];
  }

  async subscribe_webhook(data: WebhookSubscription): Promise<WebhookSubscription> {
    return this._request<WebhookSubscription>('POST', '/webhooks/subscribe', data);
  }

  async dispatch_webhook(data: WebhookDispatchRequest): Promise<WebhookDispatchResult> {
    return this._request<WebhookDispatchResult>('POST', '/webhooks/dispatch', data);
  }
}
