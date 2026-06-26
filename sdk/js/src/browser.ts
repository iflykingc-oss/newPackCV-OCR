/**
 * 浏览器专用客户端 (使用 fetch API)
 * @module browser
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
  Scenario,
  Provider,
} from './types';

export interface BrowserPackCVOptions {
  api_key: string;
  base_url?: string;
  timeout?: number;
  max_retries?: number;
  /** 自定义 fetch 实现（用于 polyfill） */
  fetch?: typeof fetch;
}

const DEFAULT_BASE_URL = 'https://api.packcv-ocr.io';
const DEFAULT_TIMEOUT = 60000;

export class BrowserPackCVClient {
  public readonly api_key: string;
  public readonly base_url: string;
  public readonly timeout: number;
  public readonly max_retries: number;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private fetch_fn: any;

  constructor(options: BrowserPackCVOptions) {
    this.api_key = options.api_key;
    this.base_url = (options.base_url ?? DEFAULT_BASE_URL).replace(/\/$/, '');
    this.timeout = options.timeout ?? DEFAULT_TIMEOUT;
    this.max_retries = options.max_retries ?? 3;
    this.fetch_fn = options.fetch ?? ((globalThis as { fetch?: typeof fetch }).fetch ?? null);
    if (!this.fetch_fn) {
      throw new PackCVError('No fetch implementation available in current environment', 0, 'no_fetch');
    }
  }

  private async _request<T>(method: string, path: string, body?: unknown): Promise<T> {
    const url = `${this.base_url}${path}`;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeout);

    let res: Response;
    try {
      res = await this.fetch_fn(url, {
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

  async batch_extract(items: BatchItem[]): Promise<BatchResult> {
    return this._request<BatchResult>('POST', '/api/v1/batch', { items });
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

  async list_api_keys(tenant_id: string): Promise<ApiKey[]> {
    const data = await this._request<ApiKey[] | { keys?: ApiKey[] }>('GET', `/api/v1/admin/tenants/${tenant_id}/api-keys`);
    if (Array.isArray(data)) return data;
    return (data?.keys ?? []) as ApiKey[];
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
}
