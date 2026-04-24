-- 多平台OCR包装识别系统 - 数据库Schema
-- 使用PostgreSQL数据库

-- ========================================
-- 1. 用户表
-- ========================================
CREATE TABLE IF NOT EXISTS users (
    id BIGSERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    phone VARCHAR(20),
    password_hash VARCHAR(255) NOT NULL,
    nickname VARCHAR(100),
    avatar_url VARCHAR(500),
    status VARCHAR(20) DEFAULT 'active', -- active, inactive, banned
    
    -- 账户类型
    account_type VARCHAR(20) DEFAULT 'personal', -- personal, enterprise
    
    -- 时间戳
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login_at TIMESTAMP,
    
    -- 索引
    CONSTRAINT valid_status CHECK (status IN ('active', 'inactive', 'banned')),
    CONSTRAINT valid_account_type CHECK (account_type IN ('personal', 'enterprise'))
);

CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_phone ON users(phone);
CREATE INDEX idx_users_status ON users(status);

-- ========================================
-- 2. 角色表
-- ========================================
CREATE TABLE IF NOT EXISTS roles (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50) UNIQUE NOT NULL,
    description VARCHAR(200),
    permissions JSONB, -- 权限列表
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 预设角色
INSERT INTO roles (name, description, permissions) VALUES
('admin', '系统管理员', ['*']::jsonb),
('user', '普通用户', ['ocr_recognize', 'model_extract', 'result_export']::jsonb),
('enterprise', '企业用户', ['ocr_recognize', 'model_extract', 'result_export', 'batch_process', 'team_manage']::jsonb) ON CONFLICT DO NOTHING;

-- ========================================
-- 3. 用户角色关联表
-- ========================================
CREATE TABLE IF NOT EXISTS user_roles (
    user_id BIGINT REFERENCES users(id) ON DELETE CASCADE,
    role_id INTEGER REFERENCES roles(id) ON DELETE CASCADE,
    assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, role_id)
);

-- ========================================
-- 4. 企业团队表
-- ========================================
CREATE TABLE IF NOT EXISTS teams (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description VARCHAR(500),
    owner_id BIGINT REFERENCES users(id),
    member_limit INTEGER DEFAULT 100,
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ========================================
-- 5. 团队成员表
-- ========================================
CREATE TABLE IF NOT EXISTS team_members (
    id SERIAL PRIMARY KEY,
    team_id INTEGER REFERENCES teams(id) ON DELETE CASCADE,
    user_id BIGINT REFERENCES users(id) ON DELETE CASCADE,
    role VARCHAR(20) DEFAULT 'member', -- owner, admin, member
    joined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(team_id, user_id)
);

-- ========================================
-- 6. OCR识别记录表
-- ========================================
CREATE TABLE IF NOT EXISTS ocr_records (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT REFERENCES users(id) ON DELETE SET NULL,
    team_id INTEGER REFERENCES teams(id) ON DELETE SET NULL,
    
    -- 输入信息
    image_url VARCHAR(500) NOT NULL,
    image_file_name VARCHAR(200),
    image_size INTEGER, -- bytes
    image_width INTEGER,
    image_height INTEGER,
    
    -- OCR引擎配置
    ocr_engine_type VARCHAR(20) NOT NULL, -- builtin, api
    ocr_engine_name VARCHAR(50),
    ocr_api_config_id INTEGER,
    
    -- OCR结果
    ocr_text TEXT,
    ocr_confidence DECIMAL(5,2),
    ocr_regions JSONB,
    processing_time DECIMAL(10,2), -- seconds
    
    -- 模型调用配置
    model_type VARCHAR(20), -- extract, correct, qa
    model_name VARCHAR(50),
    
    -- 模型结果
    structured_data JSONB,
    corrected_text TEXT,
    qa_answer TEXT,
    
    -- 输出配置
    export_format VARCHAR(10), -- json, excel, pdf
    export_file_url VARCHAR(500),
    
    -- 平台推送
    platform VARCHAR(20), -- wechat, feishu, none
    push_status VARCHAR(20), -- pending, success, failed
    push_result JSONB,
    
    -- 状态和元数据
    status VARCHAR(20) DEFAULT 'success', -- success, failed, processing
    error_message TEXT,
    metadata JSONB, -- 额外信息
    
    -- 时间戳
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_ocr_records_user_id ON ocr_records(user_id);
CREATE INDEX idx_ocr_records_team_id ON ocr_records(team_id);
CREATE INDEX idx_ocr_records_created_at ON ocr_records(created_at DESC);
CREATE INDEX idx_ocr_records_status ON ocr_records(status);

-- ========================================
-- 7. 用户配置表
-- ========================================
CREATE TABLE IF NOT EXISTS user_configs (
    id SERIAL PRIMARY KEY,
    user_id BIGINT REFERENCES users(id) ON DELETE CASCADE,
    config_key VARCHAR(100) NOT NULL,
    config_value JSONB NOT NULL,
    description VARCHAR(200),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, config_key)
);

-- ========================================
-- 8. 多平台集成配置表
-- ========================================
CREATE TABLE IF NOT EXISTS platform_integrations (
    id SERIAL PRIMARY KEY,
    user_id BIGINT REFERENCES users(id) ON DELETE CASCADE,
    team_id INTEGER REFERENCES teams(id) ON DELETE CASCADE,
    
    platform VARCHAR(20) NOT NULL, -- wechat, feishu
    integration_type VARCHAR(50) NOT NULL, -- webhook, api_key, oauth
    
    -- 集成配置（加密存储）
    credentials JSONB NOT NULL, -- webhook_url, api_key, app_id等
    is_enabled BOOLEAN DEFAULT true,
    
    -- 元数据
    metadata JSONB,
    last_used_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_platform_integrations_user_id ON platform_integrations(user_id);
CREATE INDEX idx_platform_integrations_platform ON platform_integrations(platform);

-- ========================================
-- 9. 批量处理任务表
-- ========================================
CREATE TABLE IF NOT EXISTS batch_tasks (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT REFERENCES users(id) ON DELETE SET NULL,
    team_id INTEGER REFERENCES teams(id) ON DELETE SET NULL,
    
    task_name VARCHAR(200),
    status VARCHAR(20) DEFAULT 'pending', -- pending, processing, completed, failed
    
    -- 输入配置
    image_urls JSONB NOT NULL, -- 图片URL列表
    total_count INTEGER NOT NULL,
    processed_count INTEGER DEFAULT 0,
    success_count INTEGER DEFAULT 0,
    failed_count INTEGER DEFAULT 0,
    
    -- OCR配置
    ocr_engine_type VARCHAR(20),
    ocr_api_config_id INTEGER,
    
    -- 模型配置
    model_type VARCHAR(20),
    model_name VARCHAR(50),
    
    -- 输出配置
    export_format VARCHAR(10),
    merged_export_url VARCHAR(500),
    
    -- 错误信息
    error_message TEXT,
    errors JSONB, -- 各图片的错误信息
    
    -- 时间戳
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_batch_tasks_user_id ON batch_tasks(user_id);
CREATE INDEX idx_batch_tasks_status ON batch_tasks(status);
CREATE INDEX idx_batch_tasks_created_at ON batch_tasks(created_at DESC);

-- ========================================
-- 10. 批量处理结果详情表
-- ========================================
CREATE TABLE IF NOT EXISTS batch_task_results (
    id BIGSERIAL PRIMARY KEY,
    batch_task_id BIGINT REFERENCES batch_tasks(id) ON DELETE CASCADE,
    
    image_url VARCHAR(500) NOT NULL,
    image_index INTEGER NOT NULL,
    status VARCHAR(20), -- success, failed
    
    -- OCR结果
    ocr_text TEXT,
    ocr_confidence DECIMAL(5,2),
    
    -- 模型结果
    structured_data JSONB,
    
    -- 错误信息
    error_message TEXT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_batch_task_results_batch_task_id ON batch_task_results(batch_task_id);

-- ========================================
-- 11. 模型配置表（用户自定义）
-- ========================================
CREATE TABLE IF NOT EXISTS model_configs (
    id SERIAL PRIMARY KEY,
    user_id BIGINT REFERENCES users(id) ON DELETE CASCADE,
    team_id INTEGER REFERENCES teams(id) ON DELETE SET NULL,
    
    name VARCHAR(100) NOT NULL,
    model_type VARCHAR(20) NOT NULL, -- extract, correct, qa
    model_name VARCHAR(50) NOT NULL,
    
    -- 模型参数
    temperature DECIMAL(3,2),
    max_tokens INTEGER,
    top_p DECIMAL(3,2),
    
    -- 提示词配置
    system_prompt TEXT,
    user_prompt_template TEXT,
    
    -- 其他配置
    metadata JSONB,
    is_default BOOLEAN DEFAULT false,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_model_configs_user_id ON model_configs(user_id);

-- ========================================
-- 12. OCR API配置表（用户自定义）
-- ========================================
CREATE TABLE IF NOT EXISTS ocr_api_configs (
    id SERIAL PRIMARY KEY,
    user_id BIGINT REFERENCES users(id) ON DELETE CASCADE,
    team_id INTEGER REFERENCES teams(id) ON DELETE SET NULL,
    
    name VARCHAR(100) NOT NULL,
    api_url VARCHAR(500) NOT NULL,
    api_key VARCHAR(500),
    
    -- API配置
    headers JSONB,
    parameters JSONB,
    
    -- 元数据
    description VARCHAR(500),
    is_default BOOLEAN DEFAULT false,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_ocr_api_configs_user_id ON ocr_api_configs(user_id);

-- ========================================
-- 触发器：自动更新updated_at字段
-- ========================================
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- 为需要的表添加触发器
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_teams_updated_at BEFORE UPDATE ON teams
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_ocr_records_updated_at BEFORE UPDATE ON ocr_records
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_configs_updated_at BEFORE UPDATE ON user_configs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_platform_integrations_updated_at BEFORE UPDATE ON platform_integrations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_batch_tasks_updated_at BEFORE UPDATE ON batch_tasks
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_model_configs_updated_at BEFORE UPDATE ON model_configs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_ocr_api_configs_updated_at BEFORE UPDATE ON ocr_api_configs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
