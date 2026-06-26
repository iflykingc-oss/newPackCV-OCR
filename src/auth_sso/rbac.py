"""RBAC 权限模型 - 角色 + 权限 + 租户级"""
import time
import uuid
from enum import Enum
from typing import Set, Dict, List, Optional
from pydantic import BaseModel, Field


class Permission(str, Enum):
    """细粒度权限"""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
    BILLING = "billing"
    AUDIT = "audit"
    WEBHOOK = "webhook"
    ANALYTICS = "analytics"
    APIKEY_MANAGE = "apikey:manage"
    USER_INVITE = "user:invite"


class Role(BaseModel):
    """角色"""
    role_id: str = Field(default_factory=lambda: f"role-{uuid.uuid4().hex[:8]}")
    name: str
    description: str = ""
    permissions: Set[Permission] = Field(default_factory=set)
    tenant_id: Optional[str] = None  # None 表示全局
    created_at: float = Field(default_factory=time.time)


# 预置角色模板
ROLE_TEMPLATES: Dict[str, List[Permission]] = {
    "viewer": [Permission.READ, Permission.ANALYTICS],
    "developer": [
        Permission.READ, Permission.WRITE,
        Permission.WEBHOOK, Permission.APIKEY_MANAGE,
    ],
    "billing_admin": [
        Permission.READ, Permission.BILLING, Permission.AUDIT,
    ],
    "tenant_admin": [
        Permission.READ, Permission.WRITE, Permission.DELETE,
        Permission.ADMIN, Permission.BILLING, Permission.AUDIT,
        Permission.WEBHOOK, Permission.ANALYTICS,
        Permission.APIKEY_MANAGE, Permission.USER_INVITE,
    ],
    "super_admin": list(Permission),  # 所有权限
}


class RBACManager:
    """RBAC 管理器"""

    def __init__(self):
        self._roles: Dict[str, Role] = {}
        self._user_roles: Dict[str, Set[str]] = {}  # user_id -> set of role_ids
        self._init_default_roles()

    def _init_default_roles(self) -> None:
        """初始化预置角色"""
        for name, perms in ROLE_TEMPLATES.items():
            self.create_role(name, permissions=perms, description=f"内置 {name} 角色")

    def create_role(
        self,
        name: str,
        permissions: Optional[List[Permission]] = None,
        description: str = "",
        tenant_id: Optional[str] = None,
    ) -> Role:
        """创建角色"""
        role = Role(
            name=name,
            description=description,
            permissions=set(permissions or []),
            tenant_id=tenant_id,
        )
        self._roles[role.role_id] = role
        return role

    def assign_role(self, user_id: str, role_id: str) -> bool:
        """给用户分配角色"""
        if role_id not in self._roles:
            return False
        if user_id not in self._user_roles:
            self._user_roles[user_id] = set()
        self._user_roles[user_id].add(role_id)
        return True

    def revoke_role(self, user_id: str, role_id: str) -> bool:
        """撤销角色"""
        roles = self._user_roles.get(user_id)
        if not roles or role_id not in roles:
            return False
        roles.discard(role_id)
        return True

    def get_user_permissions(
        self, user_id: str, tenant_id: Optional[str] = None
    ) -> Set[Permission]:
        """获取用户所有权限"""
        perms: Set[Permission] = set()
        for rid in self._user_roles.get(user_id, set()):
            role = self._roles.get(rid)
            if not role:
                continue
            # 角色作用域匹配：全局角色（tenant_id=None）或同租户
            if role.tenant_id is None or role.tenant_id == tenant_id:
                perms.update(role.permissions)
        return perms

    def check_permission(
        self, user_id: str, perm: Permission, tenant_id: Optional[str] = None
    ) -> bool:
        """检查权限"""
        return perm in self.get_user_permissions(user_id, tenant_id)

    def list_roles(self, tenant_id: Optional[str] = None) -> List[Role]:
        """列出角色"""
        return [
            r for r in self._roles.values()
            if tenant_id is None or r.tenant_id is None or r.tenant_id == tenant_id
        ]

    def list_user_roles(self, user_id: str) -> List[Role]:
        """列出用户角色"""
        return [self._roles[rid] for rid in self._user_roles.get(user_id, set())]
