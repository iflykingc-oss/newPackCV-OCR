# -*- coding: utf-8 -*-
"""
缓存封装
支持Redis/内存缓存
"""

import os
import json
import time
import logging
from typing import Optional, Any, Dict
from functools import wraps

logger = logging.getLogger(__name__)


class CacheManager:
    """缓存管理器"""

    def __init__(
        self,
        host: str = None,
        port: int = 6379,
        password: str = None,
        db: int = 0,
        prefix: str = "packcv:"
    ):
        """
        初始化缓存管理器

        Args:
            host: Redis主机
            port: 端口
            password: 密码
            db: 数据库编号
            prefix: 键前缀
        """
        self.host = host or os.getenv("REDIS_HOST", "localhost")
        self.port = port or int(os.getenv("REDIS_PORT", "6379"))
        self.password = password or os.getenv("REDIS_PASSWORD", "")
        self.db = db
        self.prefix = prefix

        self._client = None
        self._memory_cache: Dict[str, tuple] = {}  # key -> (value, expire_time)
        self._init_client()

    def _init_client(self):
        """初始化Redis客户端"""
        try:
            import redis
            self._client = redis.Redis(
                host=self.host,
                port=self.port,
                password=self.password or None,
                db=self.db,
                decode_responses=True
            )
            # 测试连接
            self._client.ping()
            logger.info(f"Redis缓存连接成功: {self.host}:{self.port}")
        except ImportError:
            logger.warning("Redis未安装，使用内存缓存")
            self._client = None
        except Exception as e:
            logger.warning(f"Redis连接失败: {e}，使用内存缓存")
            self._client = None

    def _make_key(self, key: str) -> str:
        """生成带前缀的键"""
        return f"{self.prefix}{key}"

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取缓存值

        Args:
            key: 缓存键
            default: 默认值

        Returns:
            缓存值或默认值
        """
        cache_key = self._make_key(key)

        if self._client:
            try:
                value = self._client.get(cache_key)
                if value is not None:
                    return json.loads(value)
            except Exception as e:
                logger.error(f"Redis get失败: {e}")

        # 回退到内存缓存
        if cache_key in self._memory_cache:
            value, expire_time = self._memory_cache[cache_key]
            if expire_time is None or expire_time > time.time():
                return json.loads(value) if isinstance(value, str) else value
            else:
                del self._memory_cache[cache_key]

        return default

    def set(
        self,
        key: str,
        value: Any,
        expire: Optional[int] = None
    ) -> bool:
        """
        设置缓存值

        Args:
            key: 缓存键
            value: 缓存值
            expire: 过期时间（秒），None表示永不过期

        Returns:
            是否设置成功
        """
        cache_key = self._make_key(key)
        serialized = json.dumps(value, ensure_ascii=False)

        if self._client:
            try:
                if expire:
                    self._client.setex(cache_key, expire, serialized)
                else:
                    self._client.set(cache_key, serialized)
                return True
            except Exception as e:
                logger.error(f"Redis set失败: {e}")

        # 回退到内存缓存
        expire_time = time.time() + expire if expire else None
        self._memory_cache[cache_key] = (serialized, expire_time)
        return True

    def delete(self, key: str) -> bool:
        """删除缓存"""
        cache_key = self._make_key(key)

        if self._client:
            try:
                self._client.delete(cache_key)
            except Exception as e:
                logger.error(f"Redis delete失败: {e}")

        if cache_key in self._memory_cache:
            del self._memory_cache[cache_key]

        return True

    def clear(self, pattern: str = "*") -> int:
        """
        清除匹配的缓存

        Args:
            pattern: 匹配模式

        Returns:
            清除的键数量
        """
        count = 0
        full_pattern = self._make_key(pattern)

        if self._client:
            try:
                keys = self._client.keys(full_pattern)
                if keys:
                    count = self._client.delete(*keys)
            except Exception as e:
                logger.error(f"Redis clear失败: {e}")

        # 清除内存缓存
        keys_to_delete = [k for k in self._memory_cache.keys()
                          if self._matches(k, full_pattern)]
        for k in keys_to_delete:
            del self._memory_cache[k]
            count += 1

        return count

    def _matches(self, key: str, pattern: str) -> bool:
        """简单的模式匹配"""
        import re
        pattern = pattern.replace('*', '.*')
        return bool(re.match(pattern, key))

    def exists(self, key: str) -> bool:
        """检查键是否存在"""
        cache_key = self._make_key(key)

        if self._client:
            try:
                return bool(self._client.exists(cache_key))
            except Exception as e:
                logger.error(f"Redis exists失败: {e}")

        return cache_key in self._memory_cache

    def expire(self, key: str, expire: int) -> bool:
        """设置过期时间"""
        cache_key = self._make_key(key)

        if self._client:
            try:
                return bool(self._client.expire(cache_key, expire))
            except Exception as e:
                logger.error(f"Redis expire失败: {e}")

        # 内存缓存
        if cache_key in self._memory_cache:
            value, _ = self._memory_cache[cache_key]
            self._memory_cache[cache_key] = (value, time.time() + expire)
            return True

        return False

    def incr(self, key: str, amount: int = 1) -> int:
        """递增计数器"""
        cache_key = self._make_key(key)

        if self._client:
            try:
                return self._client.incrby(cache_key, amount)
            except Exception as e:
                logger.error(f"Redis incr失败: {e}")

        # 内存缓存
        current = self.get(key, 0)
        new_value = current + amount
        self.set(key, new_value)
        return new_value

    def get_many(self, keys: list) -> Dict[str, Any]:
        """批量获取"""
        result = {}
        for key in keys:
            result[key] = self.get(key)
        return result

    def set_many(self, mapping: Dict[str, Any], expire: Optional[int] = None):
        """批量设置"""
        for key, value in mapping.items():
            self.set(key, value, expire)


# 全局单例
_cache_instance: Optional[CacheManager] = None


def get_cache() -> CacheManager:
    """获取缓存管理器单例"""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = CacheManager()
    return _cache_instance


def cached(key: str, expire: Optional[int] = 3600):
    """
    缓存装饰器

    Args:
        key: 缓存键（支持格式化，如 "user:{user_id}"）
        expire: 过期时间（秒）

    Usage:
        @cached("ocr:{image_hash}", expire=3600)
        def recognize_image(image_hash: str) -> dict:
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 生成缓存键
            cache_key = key.format(*args, **kwargs)

            # 尝试从缓存获取
            cache = get_cache()
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                return cached_value

            # 执行函数
            result = func(*args, **kwargs)

            # 存入缓存
            if result is not None:
                cache.set(cache_key, result, expire)

            return result
        return wrapper
    return decorator
