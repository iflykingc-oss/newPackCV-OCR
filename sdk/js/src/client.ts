/**
 * PackCV 同步客户端 (Node.js)
 * @module client
 */

import { errorFromStatus, PackCVError, NetworkError } from './exceptions';
import { retry } from './utils';
import type {
  ExtractRequest,
  ExtractResponse,
  ExtractOptions,
  BatchItem,
  BatchResult,
  QAResponse,
  UsageStats,
  Tenant,
  TenantCreate,
  ApiKey,
  HealthStatus,
  Provider,
  ProviderHealth,
  Scenario,
} from './types';

export interface PackCVClientOptions {
  api_key: string;
  base_url?: string;
  timeout?: number;
  max_retries?: number;
  user_agent?: string;
}

const DEFAULT_BASE_URL = 'https://api.packcv-ocr.io';
const DEFAULT_TIMEOUT = 60000;

// 抽象的 HTTP 客户端接口（Node.js / Browser 都可实现）
interface HttpClient {
  request(method: string, url: string, headers: Record<string, string>, body: string | null, timeout: number): Promise<HttpResponse>;
}

export interface HttpResponse {
  status: number;
  data: unknown;
}

// Node.js 实现
class NodeHttpClient implements HttpClient {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private node_fetch: any;

  constructor() {
    // 动态加载 node-fetch，避免硬依赖（浏览器端编译时不报错）
    // eslint-disable-next-line @typescript-eslint/no-implied-eval
    this.node_fetch = new Function('return require("node-fetch")')();
  }

  async request(method: string, url: string, headers: Record<string, string>, body: string | null, timeout: number): Promise<HttpResponse> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeout);

    try {
      const res = await this.node_fetch(url, {
        method,
        headers,
        body,
        signal: controller.signal,
      });
      const status = res.status;
      let data: unknown;
      try {
        data = await res.json();
      } catch {
        data = {};
      }
      return { status, data };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      if (msg.includes('abort')) {
        throw new PackCVError(`Request timeout after ${timeout}ms`, 408, 'timeout');
      }
      throw new NetworkError(msg);
    } finally {
      clearTimeout(timer);
    }
  }
}

/** 帮助函数：解包 List 响应 */
function unwrapListNamed<T>(data: T[] | { tenants?: T[]; keys?: T[]; providers?: T[]; scenarios?: T[]; [k: string]: unknown }, key: string): T[] {
  if (Array.isArray(data)) return data;
  const obj = data as Record<string, T[] | undefined>;
  return obj[key] ?? obj.data ?? [];
}

/** 同步 PackCV 客户端（异步实现 + 同步接口） */
export class PackCVClient {
  public readonly api_key: string;
  public readonly base_url: string;
  public readonly timeout: number;
  public readonly max_retries: number;
  public readonly user_agent: string;
  private readonly http: HttpClient;

  constructor(options: PackCVClientOptions, http?: HttpClient) {
    this.api_key = options.api_key;
    this.base_url = (options.base_url ?? DEFAULT_BASE_URL).replace(/\/$/, '');
    this.timeout = options.timeout ?? DEFAULT_TIMEOUT;
    this.max_retries = options.max_retries ?? 3;
    this.user_agent = options.user_agent ?? 'packcv-sdk-js/1.0.0';
    this.http = http ?? new NodeHttpClient();
  }

  // ==================== 文档提取 ====================

  /** 提取文档结构化信息 */
  async extract(file_url: string, options: ExtractOptions = {}): Promise<ExtractResponse> {
    const req: ExtractRequest = { file_url, ...options };
    return this._post<ExtractResponse>('/api/v1/extract', req);
  }

  /** 文档问答 (QA) */
  async qa(file_url: string, question: string, options: Omit<ExtractOptions, 'user_question'> = {}): Promise<QAResponse> {
    return this._post<QAResponse>('/api/v1/qa', { file_url, user_question: question, ...options });
  }

  /** 批量提取 */
  async batch_extract(items: BatchItem[]): Promise<BatchResult> {
    return this._post<BatchResult>('/api/v1/batch', { items });
  }

  /** 列出支持场景 */
  async list_scenarios(): Promise<Scenario[]> {
    const data = await this._get<Scenario[] | { scenarios?: Scenario[] }>('/api/v1/scenarios');
    return unwrapListNamed<Scenario>(data, 'scenarios');
  }

  // ==================== 租户管理 ====================

  async list_tenants(): Promise<Tenant[]> {
    const data = await this._get<Tenant[] | { tenants?: Tenant[] }>('/api/v1/admin/tenants');
    return unwrapListNamed<Tenant>(data, 'tenants');
  }

  async create_tenant(data: TenantCreate): Promise<Tenant> {
    return this._post<Tenant>('/api/v1/admin/tenants', data);
  }

  async get_tenant(tenant_id: string): Promise<Tenant> {
    return this._get<Tenant>(`/api/v1/admin/tenants/${tenant_id}`);
  }

  async update_tenant(tenant_id: string, data: Partial<TenantCreate>): Promise<Tenant> {
    return this._patch<Tenant>(`/api/v1/admin/tenants/${tenant_id}`, data);
  }

  async delete_tenant(tenant_id: string): Promise<void> {
    await this._delete(`/api/v1/admin/tenants/${tenant_id}`);
  }

  // ==================== API Key ====================

  async list_api_keys(tenant_id: string): Promise<ApiKey[]> {
    const data = await this._get<ApiKey[] | { keys?: ApiKey[] }>(`/api/v1/admin/tenants/${tenant_id}/api-keys`);
    return unwrapListNamed<ApiKey>(data, 'keys');
  }

  async create_api_key(tenant_id: string, name: string, scopes: string[] = ['read', 'write']): Promise<ApiKey> {
    return this._post<ApiKey>(`/api/v1/admin/tenants/${tenant_id}/api-keys`, { name, scopes });
  }

  async revoke_api_key(tenant_id: string, key_id: string): Promise<void> {
    await this._delete(`/api/v1/admin/tenants/${tenant_id}/api-keys/${key_id}`);
  }

  // ==================== 使用统计 ====================

  async get_usage(tenant_id: string, period?: string): Promise<UsageStats> {
    const q = period ? `?period=${period}` : '';
    return this._get<UsageStats>(`/api/v1/billing/usage/${tenant_id}${q}`);
  }

  // ==================== 健康检查 ====================

  async health(): Promise<HealthStatus> {
    return this._get<HealthStatus>('/api/v1/system/health');
  }

  // ==================== Provider 管理 ====================

  async list_providers(): Promise<Provider[]> {
    const data = await this._get<Provider[] | { providers?: Provider[] }>('/providers');
    return unwrapListNamed<Provider>(data, 'providers');
  }

  async get_provider_health(name: string): Promise<ProviderHealth> {
    return this._get<ProviderHealth>(`/providers/${name}/health`);
  }

  // ==================== HTTP 底层 ====================

  private async _get<T>(path: string): Promise<T> {
    return this._request<T>('GET', path);
  }

  private async _post<T>(path: string, body: unknown): Promise<T> {
    return this._request<T>('POST', path, body);
  }

  private async _patch<T>(path: string, body: unknown): Promise<T> {
    return this._request<T>('PATCH', path, body);
  }

  private async _delete<T>(path: string): Promise<T> {
    return this._request<T>('DELETE', path);
  }

  private async _request<T>(method: string, path: string, body?: unknown): Promise<T> {
    const fn = async (): Promise<T> => (await this._do_request(method, path, body)) as T;
    return retry(fn, {
      max_retries: this.max_retries,
      base_delay: 1000,
      max_delay: 10000,
    });
  }

  private async _do_request<T = unknown>(method: string, path: string, body?: unknown): Promise<T> {
    const url = `${this.base_url}${path}`;
    const headers: Record<string, string> = {
      'Authorization': `Bearer ${this.api_key}`,
      'Content-Type': 'application/json',
      'User-Agent': this.user_agent,
    };
    const bodyStr = body !== undefined ? JSON.stringify(body) : null;

    const res = await this.http.request(method, url, headers, bodyStr, this.timeout);

    if (res.status === 204) return undefined as unknown as T;
    if (res.status >= 200 && res.status < 300) {
      return res.data as T;
    }

    const dataObj = (res.data ?? {}) as { message?: string; error?: string };
    const errMsg = dataObj.message ?? dataObj.error ?? `HTTP ${res.status}`;
    throw errorFromStatus(res.status, errMsg, res.data as Record<string, unknown>);
  }

  close(): void {
    // 同步客户端无需显式关闭
  }
}
