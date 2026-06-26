"""租户数据模型

定义多租户SaaS架构的核心数据模型：
- TenantTier: 租户等级 (FREE/BASIC/PRO/ENTERPRISE/FLAGSHIP)
- TenantStatus: 租户状态
- TenantQuota: 三级配额 (基础+弹性+突发)
- TenantModel: 完整租户模型
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class TenantTier(str, Enum):
    """租户等级"""
    FREE = "free"
    BASIC = "basic"
    PRO = "pro"
    ENTERPRISE = "enterprise"
    FLAGSHIP = "flagship"


class TenantStatus(str, Enum):
    """租户状态"""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    DELETED = "deleted"


class IsolationLevel(str, Enum):
    """隔离级别"""
    LOGICAL = "logical"    # 逻辑隔离：共享DB+tenant_id
    PHYSICAL = "physical"  # 物理隔离：独立Schema


class BillingMode(str, Enum):
    """计费模式"""
    BY_TOKEN = "by_token"   # 按Token
    BY_CALL = "by_call"     # 按调用
    PACKAGE = "package"     # 套餐
    HYBRID = "hybrid"       # 混合（推荐）


class TenantQuota(BaseModel):
    """租户配额 - 三级配额体系

    参考DeepSeek-V4多租户推理网关设计：
    1. 基础配额 = 合同承诺最小值
    2. 弹性额度 = 当前节点空闲资源的30%
    3. 突发缓冲 = 最近5分钟实际用量的110%
    """
    # 基础配额（合同承诺）
    base_rpm: int = Field(default=60, description="基础RPM（每分钟请求数）")
    base_tpm: int = Field(default=100000, description="基础TPM（每分钟Token数）")
    base_concurrent: int = Field(default=5, description="基础并发数")

    # 弹性额度（空闲资源30%）
    elastic_rpm: int = Field(default=20, description="弹性RPM")
    elastic_tpm: int = Field(default=30000, description="弹性TPM")

    # 突发缓冲（最近5分钟实际用量110%）
    burst_buffer: int = Field(default=10, description="突发缓冲RPM")
    burst_tpm_buffer: int = Field(default=10000, description="突发缓冲TPM")

    # 月度配额
    monthly_token_quota: int = Field(default=100000, description="月度Token配额")
    monthly_call_quota: int = Field(default=1000, description="月度调用配额")

    # 计费模式
    billing_mode: BillingMode = Field(default=BillingMode.HYBRID)

    # 计量
    monthly_tokens_used: int = Field(default=0, description="本月已用Token")
    monthly_calls_used: int = Field(default=0, description="本月已用调用")

    @property
    def total_rpm(self) -> int:
        """总RPM = 基础 + 弹性"""
        return self.base_rpm + self.elastic_rpm

    @property
    def total_tpm(self) -> int:
        """总TPM = 基础 + 弹性"""
        return self.base_tpm + self.elastic_tpm

    @property
    def max_burst_rpm(self) -> int:
        """最大突发RPM = 总RPM + 突发缓冲"""
        return self.total_rpm + self.burst_buffer


class TenantModel(BaseModel):
    """租户完整数据模型"""
    tenant_id: str = Field(..., description="租户ID")
    tenant_name: str = Field(..., description="租户名称")
    tier: TenantTier = Field(default=TenantTier.FREE)
    status: TenantStatus = Field(default=TenantStatus.ACTIVE)

    # 鉴权
    api_key: str = Field(..., description="API Key")
    api_secret_hash: str = Field(..., description="API Secret SHA256哈希")
    api_secret_prefix: str = Field(default="", description="API Secret前缀（用于识别）")

    # 隔离配置
    isolation_level: IsolationLevel = Field(default=IsolationLevel.LOGICAL)
    database_schema: Optional[str] = Field(default=None, description="数据库Schema名")
    redis_namespace: str = Field(..., description="Redis命名空间（隔离数据）")

    # 配额
    quota: TenantQuota = Field(default_factory=TenantQuota)

    # 模型路由
    allowed_models: List[str] = Field(default_factory=list, description="允许使用的模型列表")
    model_routing: Dict[str, str] = Field(default_factory=dict, description="模型路由策略")

    # 联系信息（脱敏存储）
    contact_email_hash: Optional[str] = Field(default=None)
    contact_name: Optional[str] = Field(default=None)

    # 审计
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    last_active_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None

    class Config:
        use_enum_values = True


class TenantCreateRequest(BaseModel):
    """创建租户请求"""
    tenant_name: str = Field(..., min_length=1, max_length=100)
    tier: TenantTier = Field(default=TenantTier.FREE)
    contact_email: str = Field(..., description="联系邮箱")
    contact_name: Optional[str] = Field(default=None)
    allowed_models: Optional[List[str]] = Field(default=None)


class TenantInfo(BaseModel):
    """租户信息（响应中暴露的字段）"""
    tenant_id: str
    tenant_name: str
    tier: str
    status: str
    isolation_level: str
    quota: Dict
    allowed_models: List[str]
    created_at: datetime
    last_active_at: Optional[datetime] = None


class UsageStats(BaseModel):
    """使用统计"""
    tenant_id: str
    period: str
    total_calls: int
    total_tokens: int
    total_cost_usd: float
    by_model: Dict[str, Dict]
    by_scenario: Dict[str, int]
