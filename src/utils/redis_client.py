"""Redis 客户端模块 (Phase 5 兼容层)

提供统一的 Redis 客户端入口,支持:
- 单机 Redis (默认)
- 集群 Redis
- 内存 Mock (降级,无 Redis 时使用)
- 异步 API

设计原则:
- 默认返回内存 mock,保证系统可启动
- 生产环境通过 REDIS_URL 环境变量启用真实 Redis
- 异步 API 与 redis-py 兼容
"""
import os
import asyncio
import logging
import time
from typing import Any, Optional, Dict, List, Union
from threading import Lock

logger = logging.getLogger(__name__)


class InMemoryRedisMock:
    """内存版 Redis Mock,用于无 Redis 时的开发/测试环境"""

    def __init__(self) -> None:
        self._store: Dict[str, str] = {}
        self._expire: Dict[str, float] = {}
        self._lock = Lock()

    def get(self, key: str) -> Optional[str]:
        with self._lock:
            self._cleanup_expired(key)
            return self._store.get(key)

    def set(self, key: str, value: str, ex: Optional[int] = None) -> bool:
        with self._lock:
            self._store[key] = str(value)
            if ex is not None:
                self._expire[key] = time.time() + ex
            return True

    def setex(self, key: str, time_sec: int, value: str) -> bool:
        return self.set(key, value, ex=time_sec)

    def delete(self, *keys: str) -> int:
        with self._lock:
            count = 0
            for key in keys:
                if key in self._store:
                    del self._store[key]
                    if key in self._expire:
                        del self._expire[key]
                    count += 1
            return count

    def exists(self, key: str) -> bool:
        with self._lock:
            self._cleanup_expired(key)
            return key in self._store

    def incr(self, key: str, amount: int = 1) -> int:
        with self._lock:
            self._cleanup_expired(key)
            current = int(self._store.get(key, "0"))
            new_value = current + amount
            self._store[key] = str(new_value)
            return new_value

    def expire(self, key: str, time_sec: int) -> bool:
        with self._lock:
            if key in self._store:
                self._expire[key] = time.time() + time_sec
                return True
            return False

    def keys(self, pattern: str = "*") -> List[str]:
        import fnmatch
        with self._lock:
            self._cleanup_all_expired()
            return [k for k in self._store.keys() if fnmatch.fnmatch(k, pattern)]

    def ping(self) -> bool:
        return True

    def info(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "redis_version": "mock-1.0.0",
                "connected_clients": 1,
                "used_memory": len(self._store) * 64,
                "total_keys": len(self._store),
            }

    def _cleanup_expired(self, key: str) -> None:
        if key in self._expire and time.time() > self._expire[key]:
            del self._store[key]
            if key in self._expire:
                del self._expire[key]

    def _cleanup_all_expired(self) -> None:
        now = time.time()
        expired_keys = [k for k, exp in self._expire.items() if now > exp]
        for key in expired_keys:
            self._store.pop(key, None)
            self._expire.pop(key, None)

    # 异步 API 兼容
    async def aget(self, key: str) -> Optional[str]:
        return self.get(key)

    async def aset(self, key: str, value: str, ex: Optional[int] = None) -> bool:
        return self.set(key, value, ex=ex)

    async def adelete(self, *keys: str) -> int:
        return self.delete(*keys)


class RedisClient:
    """Redis 客户端统一入口"""

    def __init__(self, url: Optional[str] = None) -> None:
        self.url = url or os.getenv("REDIS_URL", "")
        self._client: Optional[Any] = None
        self._is_mock = False
        self._init_client()

    def _init_client(self) -> None:
        if not self.url:
            logger.info("REDIS_URL not set, using in-memory mock")
            self._client = InMemoryRedisMock()
            self._is_mock = True
            return
        try:
            import redis  # type: ignore
            self._client = redis.Redis.from_url(self.url, decode_responses=True)
            self._client.ping()
            logger.info("Redis connected: %s", self.url.split("@")[-1])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis connection failed (%s), falling back to in-memory mock", exc)
            self._client = InMemoryRedisMock()
            self._is_mock = True

    @property
    def is_mock(self) -> bool:
        return self._is_mock

    def __getattr__(self, item: str) -> Any:
        if self._client is None:
            raise RuntimeError("Redis client not initialized")
        return getattr(self._client, item)


# 全局单例
_redis_client: Optional[RedisClient] = None


def get_redis_client() -> RedisClient:
    """获取全局 Redis 客户端 (单例)"""
    global _redis_client
    if _redis_client is None:
        _redis_client = RedisClient()
    return _redis_client


# 兼容直接 import 的用法
redis_client = get_redis_client()
