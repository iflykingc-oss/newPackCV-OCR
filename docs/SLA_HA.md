# PackCV-OCR SLA 99.9% 灾备方案

> 目标：年可用性 99.9%（≤ 8.77 小时停机），RTO < 5分钟，RPO < 1分钟

## 一、可用性目标

| 指标 | 目标 | 监控方式 |
|------|------|---------|
| **可用性 (Availability)** | 99.9% | Prometheus `up{}` |
| **错误率 (Error Rate)** | < 0.1% | `packcv_api_requests_total{status=~"5.."}` |
| **P99 延迟** | < 5s | `histogram_quantile(0.99, ...)` |
| **RTO（恢复时间）** | < 5min | 故障演练验证 |
| **RPO（数据丢失）** | < 1min | Redis AOF + PostgreSQL WAL |

## 二、多层防护

### 2.1 应用层

```
┌─────────────────────────────────────────────┐
│  Nginx (3个实例, Keepalived VIP)              │
│  ├─ SSL终结                                  │
│  ├─ 限流                                     │
│  └─ 主动健康检查                              │
└─────────────────────────────────────────────┘
                ↓
┌─────────────────────────────────────────────┐
│  API (3+ 实例, K8s Deployment)                │
│  ├─ 进程内熔断器 (Circuit Breaker)            │
│  ├─ 重试 + 退避 (Exponential Backoff)         │
│  └─ 5级Fallback链                             │
└─────────────────────────────────────────────┘
                ↓
┌─────────────────────────────────────────────┐
│  Redis Cluster (3主3从)                       │
│  ├─ Sentinel 自动故障转移                     │
│  ├─ AOF 每秒持久化                            │
│  └─ 主从复制                                  │
└─────────────────────────────────────────────┘
                ↓
┌─────────────────────────────────────────────┐
│  PostgreSQL (主从 + WAL归档)                  │
│  ├─ 流复制                                    │
│  ├─ 自动故障转移 (Patroni)                     │
│  └─ 每日全量备份 + WAL归档                     │
└─────────────────────────────────────────────┘
```

### 2.2 网络层

- **CDN**: CloudFlare 抗DDoS
- **多可用区**: 至少2个AZ部署
- **跨区域复制**: Redis + PostgreSQL 异地备份

## 三、故障转移流程

### 3.1 单实例故障

```
时间线:
  T+0s    实例A 崩溃
  T+1s    K8s liveness probe 失败
  T+30s   K8s 重启实例A
  T+60s   实例A 健康
  T+60s+  Nginx 健康检查通过，恢复流量
```

**对用户影响**: 0（其他实例接管）

### 3.2 Redis 故障

```
  T+0s    Redis Master 崩溃
  T+1s    Sentinel 检测到
  T+10s   Sentinel 提升 Slave 为 Master
  T+15s   客户端重连
  T+30s   服务恢复
```

**对用户影响**: < 30秒（可能返回缓存的限流结果）

### 3.3 整个可用区故障

```
  T+0s    AZ-1 整体不可用
  T+10s   DNS 切换到 AZ-2
  T+30s   流量切换完成
  T+5min  RTO 完成
```

**对用户影响**: < 5分钟

## 四、备份策略

### 4.1 Redis

| 类型 | 频率 | 保留 | 恢复方式 |
|------|------|------|---------|
| RDB 快照 | 每15分钟 | 24小时 | 加载 RDB |
| AOF | 每秒 | 1小时 | `redis-check-aof` + 重放 |

### 4.2 PostgreSQL

| 类型 | 频率 | 保留 | 恢复方式 |
|------|------|------|---------|
| 全量备份 | 每日 02:00 | 30天 | `pg_restore` |
| WAL 归档 | 实时 | 7天 | PITR |

### 4.3 应用数据

- 租户配置：每日全量备份
- 计费记录：实时同步到 OLAP
- 审计日志：实时同步到 S3 冷存储

## 五、灾备演练

每月 1 次：

```bash
# 1. 模拟 Redis 故障
docker stop packcv-redis-master
# 观察: 30秒内应自动恢复
# 验证: 业务无感知

# 2. 模拟 API 实例崩溃
docker kill packcv-api-1
# 观察: K8s 30秒内重启
# 验证: 健康检查200

# 3. 模拟 PostgreSQL 故障
pg_ctl -D /var/lib/pgsql/data stop -m fast
# 观察: 60秒内主从切换
# 验证: 数据一致性

# 4. 模拟整个 AZ 故障
# 切换 DNS 到备用 AZ
# 验证: 5分钟内恢复
```

## 六、告警分级

| 级别 | 触发条件 | 通知 | 响应时间 |
|------|---------|------|---------|
| **P0** | 服务完全不可用 | 电话+短信+Slack | 5分钟 |
| **P1** | 错误率 > 1% | Slack+钉钉 | 15分钟 |
| **P2** | P99延迟 > 5s | Slack | 1小时 |
| **P3** | 资源使用率 > 80% | 邮件 | 4小时 |
| **P4** | 容量预警 | 周报 | 1周 |

## 七、SLA 报告

每月生成：

```yaml
本月SLA报告:
  计划停机: 0小时
  实际停机: 0.05小时 (3分钟)
  可用性: 99.993%
  
  故障时间线:
    - 2026-01-15 14:30-14:33: Redis主从切换
    - 2026-01-22 03:00-03:05: 例行维护
  
  改进措施:
    - 优化 Redis Sentinel 切换时间
    - 增加 API 实例副本
```

## 八、应急联系

- **值班工程师**: oncall@packcv.com
- **Slack**: #packcv-incidents
- **PagerDuty**: packcv-prod

## 九、参考

- [Google SRE Book](https://sre.google/sre-book/table-of-contents/)
- [AWS Well-Architected - Reliability Pillar](https://docs.aws.amazon.com/wellarchitected/latest/reliability-pillar/welcome.html)
