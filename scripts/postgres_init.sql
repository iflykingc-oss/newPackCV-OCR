-- ============================================================
-- PackCV-OCR PostgreSQL 初始化
-- ============================================================

-- 多租户表
CREATE TABLE IF NOT EXISTS tenants (
    tenant_id          VARCHAR(64) PRIMARY KEY,
    tenant_name        VARCHAR(255) NOT NULL,
    tier               VARCHAR(32) NOT NULL DEFAULT 'free',
    status             VARCHAR(32) NOT NULL DEFAULT 'active',
    api_key            VARCHAR(128) UNIQUE NOT NULL,
    api_secret_hash    VARCHAR(255) NOT NULL,
    isolation_level    VARCHAR(32) DEFAULT 'logical',
    redis_namespace    VARCHAR(64) NOT NULL,
    allowed_models     TEXT[] DEFAULT '{}',
    monthly_quota      INTEGER DEFAULT 1000,
    contact_email      VARCHAR(255),
    created_at         TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at         TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_active_at     TIMESTAMP WITH TIME ZONE,
    metadata           JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX idx_tenants_tier ON tenants(tier);
CREATE INDEX idx_tenants_status ON tenants(status);
CREATE INDEX idx_tenants_api_key ON tenants(api_key);

-- 使用记录表
CREATE TABLE IF NOT EXISTS usage_records (
    id                BIGSERIAL PRIMARY KEY,
    tenant_id         VARCHAR(64) NOT NULL REFERENCES tenants(tenant_id),
    request_id        VARCHAR(64) NOT NULL,
    model             VARCHAR(64) NOT NULL,
    scenario          VARCHAR(64),
    input_tokens      INTEGER NOT NULL DEFAULT 0,
    output_tokens     INTEGER NOT NULL DEFAULT 0,
    cost_usd          NUMERIC(10, 6) NOT NULL DEFAULT 0,
    cost_cny          NUMERIC(10, 4) NOT NULL DEFAULT 0,
    latency_ms        INTEGER,
    success           BOOLEAN DEFAULT TRUE,
    error_message     TEXT,
    timestamp         TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_usage_tenant_time ON usage_records(tenant_id, timestamp DESC);
CREATE INDEX idx_usage_model ON usage_records(model);
CREATE INDEX idx_usage_scenario ON usage_records(scenario);

-- 审计日志表
CREATE TABLE IF NOT EXISTS audit_logs (
    id                BIGSERIAL PRIMARY KEY,
    timestamp         TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    action            VARCHAR(64) NOT NULL,
    tenant_id         VARCHAR(64),
    user_id           VARCHAR(64),
    request_id        VARCHAR(64),
    resource          VARCHAR(255),
    status            VARCHAR(32) DEFAULT 'success',
    ip_address        INET,
    user_agent        TEXT,
    metadata          JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX idx_audit_tenant_time ON audit_logs(tenant_id, timestamp DESC);
CREATE INDEX idx_audit_action ON audit_logs(action);
CREATE INDEX idx_audit_request_id ON audit_logs(request_id);

-- 账单表
CREATE TABLE IF NOT EXISTS invoices (
    invoice_id        VARCHAR(64) PRIMARY KEY,
    tenant_id         VARCHAR(64) NOT NULL REFERENCES tenants(tenant_id),
    year_month        VARCHAR(7) NOT NULL,
    tier              VARCHAR(32) NOT NULL,
    billing_mode      VARCHAR(32) NOT NULL,
    total_tokens      INTEGER DEFAULT 0,
    call_count        INTEGER DEFAULT 0,
    package_fee       NUMERIC(10, 2) DEFAULT 0,
    package_quota     INTEGER DEFAULT 0,
    overage_tokens    INTEGER DEFAULT 0,
    overage_cost      NUMERIC(10, 2) DEFAULT 0,
    total_amount      NUMERIC(10, 2) NOT NULL,
    status            VARCHAR(32) DEFAULT 'pending',
    generated_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    paid_at           TIMESTAMP WITH TIME ZONE,
    metadata          JSONB DEFAULT '{}'::jsonb,
    UNIQUE(tenant_id, year_month)
);

CREATE INDEX idx_invoices_tenant ON invoices(tenant_id);
CREATE INDEX idx_invoices_status ON invoices(status);

-- 工作流执行日志
CREATE TABLE IF NOT EXISTS workflow_executions (
    id                BIGSERIAL PRIMARY KEY,
    run_id            VARCHAR(64) UNIQUE NOT NULL,
    tenant_id         VARCHAR(64) NOT NULL,
    scenario          VARCHAR(64),
    input_file_url    TEXT,
    detected_scenario VARCHAR(64),
    model_used        VARCHAR(64),
    success           BOOLEAN DEFAULT TRUE,
    total_latency_ms  INTEGER,
    api_call_count    INTEGER DEFAULT 0,
    error_message     TEXT,
    started_at        TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at      TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_workflow_tenant_time ON workflow_executions(tenant_id, started_at DESC);
CREATE INDEX idx_workflow_scenario ON workflow_executions(scenario);

-- 演示数据（dev/test only - 生产环境会删除）
INSERT INTO tenants (tenant_id, tenant_name, tier, api_key, api_secret_hash, redis_namespace, allowed_models, monthly_quota)
VALUES
    ('tnt_demo_pro', '演示企业客户', 'pro', 'pk_test_demo_pro_001', 'hash_demo_pro_secret', 'packcv:pro', ARRAY['doubao-seed-2-0-pro-260215', 'kimi-k2-5-260127'], 1000000),
    ('tnt_demo_free', '免费体验用户', 'free', 'pk_test_demo_free_001', 'hash_demo_free_secret', 'packcv:free', ARRAY['doubao-seed-2-0-mini-260215'], 1000)
ON CONFLICT (tenant_id) DO NOTHING;
