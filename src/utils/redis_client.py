"""Redis客户端封装 - 多租户隔离存储

提供：
- 单例Redis连接池
- 租户命名空间隔离
- 常用操作封装（限流、计数、缓存）
"""

import os
from typing import Any, Optional

try:
    import redis
    from redis.connection import ConnectionPool
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None
    ConnectionPool = None


class TenantRedisClient:
    """多租户Redis客户端

    所有Key自动加上namespace前缀，实现租户间数据隔离
    """

    _instance: Optional["TenantRedisClient"] = None
    _pool: Any = None

    def __new__(cls) -> "TenantRedisClient":
        if not REDIS_AVAILABLE:
            raise ImportError("redis package not installed. Install with: uv add redis")
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._init_pool()
        return cls._instance

    @classmethod
    def _init_pool(cls) -> None:
        """初始化连接池（单例）"""
        if not REDIS_AVAILABLE:
            raise ImportError("redis package not installed")
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        max_connections = int(os.getenv("REDIS_MAX_CONNECTIONS", "50"))
        socket_timeout = float(os.getenv("REDIS_SOCKET_TIMEOUT", "5"))

        cls._pool = ConnectionPool.from_url(
            redis_url,
            max_connections=max_connections,
            socket_timeout=socket_timeout,
            socket_connect_timeout=socket_timeout,
            decode_responses=True,
        )

    @property
    def client(self) -> Any:
        """获取Redis客户端"""
        if not REDIS_AVAILABLE:
            raise ImportError("redis package not installed")
        if self._pool is None:
            self._init_pool()
        return redis.Redis(connection_pool=self._pool)

    def namespaced_key(self, namespace: str, key: str) -> str:
        """生成租户命名空间Key

        Args:
            namespace: 租户命名空间
            key: 原始Key

        Returns:
            命名空间Key，格式: packcv:{namespace}:{key}
        """
        return f"packcv:{namespace}:{key}"

    def get(self, namespace: str, key: str) -> Optional[str]:
        """读取（带命名空间）"""
        full_key = self.namespaced_key(namespace, key)
        return self.client.get(full_key)

    def set(
        self,
        namespace: str,
        key: str,
        value: Any,
        ex: Optional[int] = None,
    ) -> bool:
        """写入（带命名空间）

        Args:
            namespace: 租户命名空间
            key: 键
            value: 值（自动str转换）
            ex: 过期时间（秒）
        """
        full_key = self.namespaced_key(namespace, key)
        return bool(self.client.set(full_key, str(value), ex=ex))

    def incr(self, namespace: str, key: str, amount: int = 1) -> int:
        """自增（带命名空间）"""
        full_key = self.namespaced_key(namespace, key)
        return int(self.client.incrby(full_key, amount))

    def hincrby(
        self, namespace: str, key: str, field: str, amount: int = 1
    ) -> int:
        """Hash自增"""
        full_key = self.namespaced_key(namespace, key)
        return int(self.client.hincrby(full_key, field, amount))

    def hincrbyfloat(
        self, namespace: str, key: str, field: str, amount: float
    ) -> float:
        """Hash自增浮点"""
        full_key = self.namespaced_key(namespace, key)
        return float(self.client.hincrbyfloat(full_key, field, amount))

    def hgetall(self, namespace: str, key: str) -> dict:
        """读取Hash全部"""
        full_key = self.namespaced_key(namespace, key)
        return self.client.hgetall(full_key)

    def delete(self, namespace: str, *keys: str) -> int:
        """删除（带命名空间）"""
        full_keys = [self.namespaced_key(namespace, k) for k in keys]
        return int(self.client.delete(*full_keys))

    def expire(self, namespace: str, key: str, seconds: int) -> bool:
        """设置过期"""
        full_key = self.namespaced_key(namespace, key)
        return bool(self.client.expire(full_key, seconds))

    def exists(self, namespace: str, key: str) -> bool:
        """判断Key是否存在"""
        full_key = self.namespaced_key(namespace, key)
        return bool(self.client.exists(full_key))

    def eval_script(
        self, script: str, namespace: str, num_keys: int, *args
    ) -> Any:
        """执行Lua脚本（自动加namespace前缀到KEYS）"""
        full_keys = [
            self.namespaced_key(namespace, k) for k in args[:num_keys]
        ]
        # 剩余的是ARGV
        argv = args[num_keys:]
        return self.client.eval(script, num_keys, *full_keys, *argv)

    def load_script(self, script: str) -> str:
        """加载Lua脚本并返回SHA"""
        return self.client.script_load(script)

    def evalsha(
        self, sha: str, namespace: str, num_keys: int, *args
    ) -> Any:
        """通过SHA执行Lua脚本"""
        full_keys = [
            self.namespaced_key(namespace, k) for k in args[:num_keys]
        ]
        argv = args[num_keys:]
        return self.client.evalsha(sha, num_keys, *full_keys, *argv)

    def health_check(self) -> bool:
        """健康检查"""
        try:
            return bool(self.client.ping())
        except Exception:
            return False


# 全局单例
try:
    redis_client = TenantRedisClient()
except ImportError:
    # redis未安装, 创建None占位; 使用方应检查REDIS_AVAILABLE
    redis_client = None
