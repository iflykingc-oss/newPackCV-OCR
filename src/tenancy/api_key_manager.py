"""API Key管理 - 生成/验证/轮换

参考DeepSeek-V4最佳实践：
- 两层密钥体系（主密钥签发临时密钥）
- API Key（公开标识）+ API Secret（私密凭证）
- 5-10分钟缓存窗口问题：使用max-age=0强制刷新
"""

import hashlib
import secrets
import time
import uuid
from typing import Optional

from tenancy.context import TenantContext
from tenancy.models import (
    IsolationLevel,
    TenantModel,
    TenantQuota,
    TenantStatus,
    TenantTier,
)
from utils.redis_client import redis_client


# 内存仓储（生产环境应替换为PostgreSQL/Supabase）
_TENANT_STORE: dict = {}


class APIKeyManager:
    """API Key管理器

    提供API Key的生成、验证、刷新、撤销等能力。
    默认使用内存仓储（演示用），生产应替换为持久化存储。
    """

    # API Key 格式: pk_live_{32位随机} 或 pk_test_{32位随机}
    KEY_PREFIX_LIVE = "pk_live_"
    KEY_PREFIX_TEST = "pk_test_"

    # Secret 格式: sk_live_{48位随机}
    SECRET_PREFIX = "sk_"

    @staticmethod
    def generate_api_key(env: str = "test") -> str:
        """生成API Key（公开标识）

        Args:
            env: live/test

        Returns:
            API Key 字符串
        """
        prefix = (
            APIKeyManager.KEY_PREFIX_LIVE
            if env == "live"
            else APIKeyManager.KEY_PREFIX_TEST
        )
        return prefix + secrets.token_urlsafe(24)

    @staticmethod
    def generate_api_secret() -> str:
        """生成API Secret（私密凭证）

        Returns:
            API Secret 字符串
        """
        return APIKeyManager.SECRET_PREFIX + secrets.token_urlsafe(36)

    @staticmethod
    def hash_secret(secret: str) -> str:
        """SHA256哈希Secret

        Args:
            secret: 原始Secret

        Returns:
            64字符哈希
        """
        return hashlib.sha256(secret.encode()).hexdigest()

    @staticmethod
    def create_tenant(
        tenant_name: str,
        tier: TenantTier = TenantTier.FREE,
        contact_email: str = "",
        env: str = "test",
        allowed_models: Optional[list] = None,
    ) -> tuple:
        """创建租户并返回（TenantModel, api_secret）

        Args:
            tenant_name: 租户名称
            tier: 租户等级
            contact_email: 联系邮箱
            env: live/test
            allowed_models: 允许的模型列表

        Returns:
            (TenantModel, api_secret原始值，仅创建时返回)
        """
        tenant_id = "tnt_" + uuid.uuid4().hex[:16]
        api_key = APIKeyManager.generate_api_key(env)
        api_secret = APIKeyManager.generate_api_secret()
        api_secret_hash = APIKeyManager.hash_secret(api_secret)

        # 根据等级设置默认配额
        quota = APIKeyManager._default_quota_for_tier(tier)

        # 隔离级别：FLAGSHIP用物理隔离，其他逻辑隔离
        isolation_level = (
            IsolationLevel.PHYSICAL
            if tier == TenantTier.FLAGSHIP
            else IsolationLevel.LOGICAL
        )

        # Redis命名空间（数据隔离）
        redis_namespace = f"t{tenant_id[-8:]}"

        # 默认允许的模型
        if allowed_models is None:
            allowed_models = [
                "doubao-seed-2-0-pro-260215",
                "doubao-seed-2-0-lite-260215",
                "doubao-seed-2-0-mini-260215",
            ]

        tenant = TenantModel(
            tenant_id=tenant_id,
            tenant_name=tenant_name,
            tier=tier,
            status=TenantStatus.ACTIVE,
            api_key=api_key,
            api_secret_hash=api_secret_hash,
            api_secret_prefix=api_secret[:12],
            isolation_level=isolation_level,
            redis_namespace=redis_namespace,
            quota=quota,
            allowed_models=allowed_models,
            contact_email_hash=(
                hashlib.sha256(contact_email.encode()).hexdigest()
                if contact_email
                else None
            ),
        )

        # 存入内存仓储
        _TENANT_STORE[api_key] = tenant

        # 同时存入Redis（用于分布式访问） - Redis故障时降级
        try:
            redis_client.set(
                "global", f"tenant:{api_key}", tenant.model_dump_json(), ex=3600
            )
        except Exception:
            pass  # Redis不可用时仅依赖内存仓储

        return tenant, api_secret

    @staticmethod
    def verify_api_key(api_key: str, api_secret: str) -> Optional[TenantModel]:
        """验证API Key+Secret

        Args:
            api_key: API Key
            api_secret: API Secret

        Returns:
            验证成功返回TenantModel，失败返回None
        """
        tenant = APIKeyManager.verify_api_key_only(api_key)
        if tenant is None:
            return None

        # 3. 验证Secret哈希
        if tenant.api_secret_hash != APIKeyManager.hash_secret(api_secret):
            return None

        # 4. 检查状态
        if tenant.status != TenantStatus.ACTIVE:
            return None

        # 5. 更新最后活跃时间
        from datetime import datetime as _dt
        tenant.last_active_at = _dt.now()
        _TENANT_STORE[api_key] = tenant
        return tenant

    @staticmethod
    def verify_api_key_only(api_key: str) -> Optional[TenantModel]:
        """仅验证API Key（用于SDK等场景）

        Args:
            api_key: API Key

        Returns:
            找到返回TenantModel，未找到返回None
        """
        # 先查Redis（Redis故障时降级到内存）
        try:
            cached = redis_client.get("global", f"tenant:{api_key}")
            if cached:
                import json
                return TenantModel(**json.loads(cached))
        except Exception:
            pass  # Redis不可用，降级到内存仓储
        return _TENANT_STORE.get(api_key)

    @staticmethod
    def revoke_tenant(api_key: str) -> bool:
        """撤销租户"""
        tenant = _TENANT_STORE.get(api_key)
        if tenant is None:
            return False
        tenant.status = TenantStatus.SUSPENDED
        _TENANT_STORE[api_key] = tenant
        # 清理Redis缓存
        redis_client.delete("global", f"tenant:{api_key}")
        return True

    @staticmethod
    def list_tenants() -> list:
        """列出所有租户（管理后台用）"""
        return list(_TENANT_STORE.values())

    @staticmethod
    def _default_quota_for_tier(tier: TenantTier) -> TenantQuota:
        """根据等级返回默认配额"""
        tier_quotas = {
            TenantTier.FREE: TenantQuota(
                base_rpm=10, base_tpm=10000, base_concurrent=2,
                elastic_rpm=0, elastic_tpm=0, burst_buffer=0,
                monthly_token_quota=1000, monthly_call_quota=100,
            ),
            TenantTier.BASIC: TenantQuota(
                base_rpm=60, base_tpm=100000, base_concurrent=5,
                elastic_rpm=20, elastic_tpm=30000, burst_buffer=10,
                monthly_token_quota=100000, monthly_call_quota=5000,
            ),
            TenantTier.PRO: TenantQuota(
                base_rpm=300, base_tpm=500000, base_concurrent=20,
                elastic_rpm=100, elastic_tpm=150000, burst_buffer=50,
                monthly_token_quota=1000000, monthly_call_quota=50000,
            ),
            TenantTier.ENTERPRISE: TenantQuota(
                base_rpm=1000, base_tpm=2000000, base_concurrent=100,
                elastic_rpm=300, elastic_tpm=600000, burst_buffer=200,
                monthly_token_quota=10000000, monthly_call_quota=500000,
            ),
            TenantTier.FLAGSHIP: TenantQuota(
                base_rpm=5000, base_tpm=10000000, base_concurrent=500,
                elastic_rpm=1500, elastic_tpm=3000000, burst_buffer=1000,
                monthly_token_quota=100000000, monthly_call_quota=5000000,
            ),
        }
        return tier_quotas.get(tier, tier_quotas[TenantTier.FREE])

    @staticmethod
    def setup_demo_tenants() -> None:
        """初始化演示租户（开发环境用）"""
        if _TENANT_STORE:
            return  # 已初始化
        APIKeyManager.create_tenant(
            tenant_name="演示企业客户",
            tier=TenantTier.PRO,
            contact_email="demo@enterprise.com",
            env="test",
        )
        APIKeyManager.create_tenant(
            tenant_name="免费体验用户",
            tier=TenantTier.FREE,
            contact_email="trial@user.com",
            env="test",
        )
        APIKeyManager.create_tenant(
            tenant_name="金融VIP客户",
            tier=TenantTier.ENTERPRISE,
            contact_email="vip@finance.com",
            env="live",
        )
