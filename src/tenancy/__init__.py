"""租户管理模块 - 多租户SaaS架构核心

提供：
- 租户数据模型 (TenantModel, TenantQuota)
- 租户上下文 (TenantContext - 请求级别隔离)
- API Key管理 (生成/验证/轮换)
- 租户级限流器 (滑动窗口+令牌桶)
- 租户仓储 (内存/数据库双模式)
"""
