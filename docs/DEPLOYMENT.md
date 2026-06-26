# PackCV-OCR 生产部署指南

> 适用版本：V7.0+ | 部署时间：~15分钟 | 维护成本：低

## 一、部署架构

```
                          ┌─────────────────┐
                          │   DNS/CDN        │
                          │   (CloudFlare)   │
                          └────────┬────────┘
                                   │
                          ┌────────▼─────────┐
                          │   Nginx (80/443) │
                          │   SSL终结+负载均衡│
                          │   Rate Limit     │
                          └────────┬─────────┘
                                   │
                ┌──────────────────┼──────────────────┐
                │                  │                  │
        ┌───────▼────────┐  ┌──────▼──────┐  ┌──────▼──────┐
        │  API实例1       │  │  API实例2   │  │  API实例3   │
        │  :9000 (内部)    │  │  :9000     │  │  :9000     │
        │  gunicorn+uvicorn│  │            │  │            │
        └───────┬────────┘  └──────┬──────┘  └──────┬──────┘
                │                  │                  │
        ┌───────▼──────────────────▼──────────────────▼────────┐
        │         Redis Cluster (主从)                          │
        │         限流+计费+审计+租户上下文                       │
        └────────────────────────────────────────────────────┘
                │
        ┌───────▼────────┐
        │  PostgreSQL     │  ← 可选（生产建议开启）
        │  租户持久化      │
        └────────────────┘

        ┌─────────────────────────────────────────────────────┐
        │  Prometheus ← /metrics 抓取                          │
        │  Grafana    ← 可视化                                 │
        │  AlertManager ← 告警通知 (Slack/钉钉/邮件)            │
        └─────────────────────────────────────────────────────┘
```

## 二、部署方式

### 方式 1：Docker Compose（推荐中小规模）

#### 1. 准备环境

```bash
# 克隆代码
git clone <repo-url> packcv-ocr && cd packcv-ocr

# 复制环境变量模板
cp .env.example .env
vim .env  # 修改密码、密钥等
```

#### 2. 启动基础服务

```bash
# 启动 API + Redis + Postgres
docker-compose -f docker-compose.base.yml up -d

# 查看日志
docker-compose -f docker-compose.base.yml logs -f api
```

#### 3. 启动 Nginx 反向代理

```bash
# 准备SSL证书（Let's Encrypt）
certbot certonly --standalone -d api.yourdomain.com

# 启动 Nginx
docker-compose -f docker-compose.nginx.yml up -d

# 验证
curl -I https://api.yourdomain.com/api/v1/health
```

#### 4. 启动监控（可选）

```bash
# Prometheus + Grafana + AlertManager
docker-compose -f docker-compose.monitoring.yml up -d

# 访问 Grafana: http://localhost:3000
# 账号: admin / <GRAFANA_PASSWORD>
```

#### 5. 启动生产全栈

```bash
# 一键启动（基础服务 + Nginx + 监控）
docker-compose -f docker-compose.prod.yml up -d

# 验证所有服务
./scripts/start.sh
```

### 方式 2：Kubernetes（推荐大规模）

```bash
# 1. 创建命名空间
kubectl create namespace packcv

# 2. 创建Secret
kubectl create secret generic packcv-secrets \
  --from-env-file=.env -n packcv

# 3. 部署
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml

# 4. 检查状态
kubectl get pods -n packcv
```

## 三、环境变量

| 变量 | 必填 | 示例 | 说明 |
|------|------|------|------|
| `ENV` | ✅ | `production` | 环境标识 |
| `REDIS_URL` | ✅ | `redis://redis:6379/0` | Redis地址 |
| `POSTGRES_URL` | ⚠️ | `postgresql://...` | 持久化（可选） |
| `LANGSMITH_API_KEY` | ❌ | `lsv2_xxx` | 可观测性（可选） |
| `GRAFANA_PASSWORD` | ⚠️ | `<strong-pwd>` | Grafana管理员密码 |
| `NGINX_SSL_CERT` | ⚠️ | `/etc/nginx/certs/...` | SSL证书路径 |

## 四、验证部署

```bash
# 1. 健康检查
curl https://api.yourdomain.com/api/v1/health

# 2. 创建演示租户
curl -X POST https://api.yourdomain.com/api/v1/admin/tenants/demo

# 3. 测试核心功能
curl -H "X-API-Key: <demo-key>" \
  https://api.yourdomain.com/api/v1/scenarios

# 4. 验证监控
curl https://api.yourdomain.com/metrics | head -5
```

## 五、回滚

```bash
# 1. 停止新版本
docker-compose -f docker-compose.base.yml down

# 2. 回滚到上个版本
git checkout v6.x
docker-compose -f docker-compose.base.yml up -d

# 3. 验证
./scripts/health_check.sh
```

## 六、升级

```bash
# 1. 拉取新镜像
docker pull packcv/api:v7.x

# 2. 滚动更新（零停机）
docker-compose -f docker-compose.base.yml up -d --no-deps api

# 3. 监控日志
docker-compose -f docker-compose.base.yml logs -f api
```

## 七、灾备切换

详见 `docs/SLA_HA.md`

## 八、监控告警

详见 `docs/MONITORING.md`

## 九、常见问题

### Q1: 端口冲突
修改 `docker-compose.base.yml` 中的端口映射。

### Q2: Redis 连接失败
```bash
docker exec -it packcv-redis redis-cli ping
```

### Q3: LLM 调用失败
检查 `LANGSMITH_API_KEY` 或模型API密钥。

### Q4: 性能问题
- 调整 gunicorn workers: `WEB_CONCURRENCY=8`
- 启用 Redis 集群
- 启用 PostgreSQL 连接池
