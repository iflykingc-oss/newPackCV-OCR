# PackCV-OCR 运维手册

> 日常运维操作指南

## 一、启动 / 停止

### 1.1 一键启动

```bash
./scripts/start.sh
```

自动完成：
1. 启动 Redis
2. 启动 API (3 workers)
3. 启动 Nginx
4. 启动 Prometheus + Grafana
5. 健康检查

### 1.2 一键停止

```bash
./scripts/stop.sh
```

### 1.3 重启

```bash
./scripts/stop.sh && ./scripts/start.sh
```

## 二、状态检查

### 2.1 服务状态

```bash
# Docker
docker ps | grep packcv

# 进程
ps -ef | grep uvicorn

# 端口
ss -tlnp | grep 9000
```

### 2.2 健康检查

```bash
# API
curl -I http://localhost:9000/api/v1/health

# Redis
redis-cli ping

# Prometheus
curl -I http://localhost:9090/-/healthy

# Grafana
curl -I http://localhost:3000/api/health
```

### 2.3 资源使用

```bash
# CPU/内存
docker stats --no-stream

# 磁盘
df -h

# Redis内存
redis-cli info memory
```

## 三、监控指标

### 3.1 业务指标（核心 SLO）

| 指标 | 查询 | 阈值 |
|------|------|------|
| API 错误率 | `sum(rate(packcv_api_requests_total{status=~"5.."}[5m])) / sum(rate(packcv_api_requests_total[5m]))` | < 0.1% |
| P99 延迟 | `histogram_quantile(0.99, rate(packcv_api_request_duration_seconds_bucket[5m]))` | < 5s |
| 限流命中率 | `rate(packcv_rate_limit_hits_total[5m])` | < 1% |
| 租户数 | `packcv_tenants_total` | 监控增长 |

### 3.2 系统指标

| 指标 | 阈值 |
|------|------|
| CPU 使用率 | < 70% |
| 内存使用率 | < 80% |
| 磁盘使用率 | < 85% |
| Redis 内存 | < 80% of `maxmemory` |
| 网络连接数 | < 80% of `net.core.somaxconn` |

## 四、告警处理

### 4.1 P0 告警（服务不可用）

```bash
# 1. 立即查看
./scripts/health_check.sh

# 2. 查看错误日志
docker logs --tail 100 packcv-api-1 | grep -E "ERROR|Exception"

# 3. 检查上游
curl -I https://ark.cn-beijing.volces.com/

# 4. 必要时手动切换
./scripts/failover.sh
```

### 4.2 P1 告警（错误率高）

```bash
# 1. 查看错误分布
curl -s http://localhost:9090/api/v1/query?query=sum%20by%20(status)(rate(packcv_api_requests_total{status%3D~%225..%22}[5m]))

# 2. 查看具体错误
docker logs --tail 200 packcv-api-1 | grep -B 2 "500\|503"

# 3. 是否需要降级？
curl -X POST http://localhost:9000/api/v1/admin/degradation \
  -d '{"level":"degraded"}'
```

### 4.3 P2 告警（延迟高）

```bash
# 1. 慢查询
curl -s 'http://localhost:9090/api/v1/query?query=topk(10,%20histogram_quantile(0.99,%20rate(packcv_api_request_duration_seconds_bucket[5m])))'

# 2. Redis 慢查询
redis-cli SLOWLOG GET 10

# 3. 是否需要扩容？
docker stats
```

## 五、备份与恢复

### 5.1 手动备份

```bash
# Redis
docker exec packcv-redis redis-cli BGSAVE

# PostgreSQL
docker exec packcv-postgres pg_dump -U packcv packcv > /backup/$(date +%Y%m%d).sql
```

### 5.2 恢复

```bash
# Redis
docker exec -i packcv-redis redis-cli < /backup/dump.rdb

# PostgreSQL
cat /backup/20260101.sql | docker exec -i packcv-postgres psql -U packcv packcv
```

## 六、租户管理

### 6.1 创建租户

```bash
curl -X POST http://localhost:9000/api/v1/admin/tenants \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_name": "客户A",
    "tier": "pro",
    "contact_email": "admin@customer.com"
  }'
```

### 6.2 暂停租户

```bash
curl -X POST http://localhost:9000/api/v1/admin/tenants/{api_key}/suspend
```

### 6.3 删除租户

```bash
curl -X DELETE http://localhost:9000/api/v1/admin/tenants/{api_key}
```

## 七、计费管理

### 7.1 查看账单

```bash
curl -H "X-API-Key: <key>" http://localhost:9000/api/v1/billing/invoice
```

### 7.2 调整套餐

```bash
curl -X POST http://localhost:9000/api/v1/admin/tenants/{api_key}/tier \
  -d '{"tier":"enterprise"}'
```

## 八、日志

### 8.1 应用日志

```bash
# 实时跟踪
docker logs -f packcv-api-1

# 错误过滤
docker logs --tail 1000 packcv-api-1 | grep -E "ERROR|WARN" | tail -20
```

### 8.2 审计日志

```bash
curl -H "X-API-Key: <key>" \
  "http://localhost:9000/api/v1/audit/logs?action=BILLING_DEDUCT&limit=50"
```

## 九、扩容

### 9.1 水平扩容

```bash
# K8s
kubectl scale deployment packcv-api --replicas=10

# Docker Compose
docker-compose up -d --scale api=10
```

### 9.2 垂直扩容

修改 `docker-compose.base.yml`:
```yaml
services:
  api:
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
```

## 十、故障排查清单

| 现象 | 可能原因 | 排查命令 |
|------|---------|---------|
| API 502 | Nginx 上游不可用 | `docker logs packcv-nginx` |
| Redis 超时 | 网络或 OOM | `redis-cli info stats` |
| LLM 失败 | API 配额/限流 | 检查 `LANGSMITH_API_KEY` |
| 限流误杀 | 配额配置过严 | `redis-cli hgetall quota:*` |
| 性能差 | 资源不足 | `docker stats` |
| 启动失败 | 配置错误 | `docker logs packcv-api-1` |
