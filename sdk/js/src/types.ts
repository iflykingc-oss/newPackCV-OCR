/**
 * PackCV SDK 公共类型定义
 */

/** 租户套餐等级 */
export type Tier = 'FREE' | 'BASIC' | 'PRO' | 'ENTERPRISE' | 'FLAGSHIP';

/** 文档场景 */
export type ScenarioId =
  | 'resume' | 'contract' | 'invoice' | 'id_card' | 'business_license'
  | 'bank_statement' | 'medical_record' | 'academic_paper' | 'product_manual'
  | 'legal_doc' | 'financial_report' | 'custom';

export interface Scenario {
  scenario_id: ScenarioId;
  name: string;
  description: string;
  default_fields: string[];
  supported_formats: string[];
}

/** 文档提取请求 */
export interface ExtractRequest {
  /** 文档URL（必填） */
  file_url: string;
  /** 文档场景 */
  scenario?: ScenarioId;
  /** 用户问题（QA 模式） */
  user_question?: string;
  /** 目标提取字段（覆盖默认值） */
  target_fields?: string[];
  /** LLM Provider 名称（可选，不指定走路由） */
  provider?: string;
  /** 文档语言 */
  language?: string;
  /** 自定义提示词 */
  custom_prompt?: string;
  /** 是否启用脱敏 */
  enable_masking?: boolean;
  /** 超时秒数 */
  timeout?: number;
}

export interface ExtractOptions {
  scenario?: ScenarioId;
  user_question?: string;
  target_fields?: string[];
  provider?: string;
  enable_masking?: boolean;
  timeout?: number;
}

export interface BatchItem {
  file_url: string;
  scenario?: ScenarioId;
  user_question?: string;
}

export interface BatchResult {
  total: number;
  succeeded: number;
  failed: number;
  items: Array<{
    file_url: string;
    status: 'success' | 'failed';
    data?: Record<string, unknown>;
    error?: string;
  }>;
}

/** 文档提取响应 */
export interface ExtractResponse {
  /** 唯一运行ID */
  run_id: string;
  /** 结构化数据 */
  structured_data: Record<string, unknown>;
  /** 置信度（0-1） */
  confidence: number;
  /** QA 答案（如果提供了 user_question） */
  qa_answer?: string;
  /** 使用的场景 */
  scenario: ScenarioId;
  /** 验证错误 */
  validation_errors: string[];
  /** 文档语言 */
  language?: string;
  /** Token 消耗 */
  tokens_used?: {
    input: number;
    output: number;
    total: number;
  };
  /** 耗时（毫秒） */
  duration_ms: number;
  /** 使用的 Provider */
  provider_used?: string;
}

/** QA 响应 */
export interface QAResponse {
  run_id: string;
  question: string;
  answer: string;
  confidence: number;
  citations?: Array<{
    source: string;
    content: string;
    relevance: number;
  }>;
}

export interface UsageByDay {
  date: string;
  calls: number;
  tokens: number;
  cost: number;
}

export interface UsageByProvider {
  provider: string;
  calls: number;
  tokens: number;
  cost: number;
}

export interface UsageStats {
  tenant_id: string;
  period: string;
  total_calls: number;
  total_tokens: number;
  total_cost: number;
  by_day: UsageByDay[];
  by_provider: UsageByProvider[];
}

/** 租户 */
export interface Tenant {
  tenant_id: string;
  name: string;
  tier: Tier;
  status: 'active' | 'suspended' | 'cancelled';
  created_at: string;
  quota: {
    max_rpm: number;
    max_tpm: number;
    max_concurrent: number;
    monthly_quota: number;
  };
}

export interface TenantCreate {
  name: string;
  tier: Tier;
  contact_email?: string;
  description?: string;
}

export interface ApiKey {
  key_id: string;
  api_key: string;
  name: string;
  created_at: string;
  expires_at?: string;
  last_used_at?: string;
  scopes: string[];
}

export interface HealthStatus {
  status: 'healthy' | 'degraded' | 'unhealthy';
  version: string;
  uptime_seconds: number;
  components: Record<string, {
    status: 'up' | 'down';
    latency_ms?: number;
    message?: string;
  }>;
}

/** LLM Provider */
export interface Provider {
  name: string;
  display_name: string;
  vendor: string;
  enabled: boolean;
  models: string[];
  priority: number;
}

export interface ProviderHealth {
  name: string;
  status: 'healthy' | 'degraded' | 'unhealthy';
  latency_ms: number;
  success_rate: number;
  last_check: string;
}

/** WebHook */
export interface WebhookSubscription {
  subscription_id: string;
  tenant_id: string;
  url: string;
  events: string[];
  secret: string;
  active: boolean;
  created_at: string;
}

export interface WebhookDispatchRequest {
  event_type: string;
  tenant_id: string;
  data: Record<string, unknown>;
}

export interface WebhookDispatchResult {
  event_id: string;
  dispatched_to: number;
  results: Array<{
    subscription_id: string;
    url: string;
    status: 'success' | 'failed';
    http_code?: number;
    error?: string;
  }>;
}
