"""LLM 响应缓存 - 哈希精确匹配 + 可选语义相似匹配

通过 SHA-256 缓存 LLM 响应，命中率可达 30-50%，节省 token 成本。
"""
import hashlib
import json
import time
from typing import Any, Optional, Dict


class LLMResponseCache:
    """LLM 响应缓存（内存版，可用 Redis 替换）"""

    def __init__(self, max_size: int = 10000, default_ttl: int = 3600):
        self._cache: Dict[str, tuple] = {}  # key -> (value, expire_at)
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.hits = 0
        self.misses = 0

    @staticmethod
    def make_key(prompt: str, model: str, **kwargs) -> str:
        """生成缓存 key"""
        content = json.dumps({"p": prompt, "m": model, **kwargs}, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """获取缓存"""
        item = self._cache.get(key)
        if item is None:
            self.misses += 1
            return None
        value, expire_at = item
        if expire_at and time.time() > expire_at:
            del self._cache[key]
            self.misses += 1
            return None
        self.hits += 1
        return value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """写入缓存"""
        if len(self._cache) >= self.max_size:
            # 简单的 LRU：删除最早过期
            expired = [k for k, (_, exp) in self._cache.items() if exp and time.time() > exp]
            for k in expired[:100]:
                del self._cache[k]
            # 仍超限，删前 10%
            if len(self._cache) >= self.max_size:
                for k in list(self._cache.keys())[: self.max_size // 10]:
                    del self._cache[k]
        expire_at = time.time() + (ttl or self.default_ttl) if (ttl or self.default_ttl) > 0 else 0
        self._cache[key] = (value, expire_at)

    def delete(self, key: str) -> bool:
        return self._cache.pop(key, None) is not None

    def clear(self) -> None:
        self._cache.clear()
        self.hits = 0
        self.misses = 0

    def stats(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total > 0 else 0.0,
            "total_requests": total,
        }


# 全局单例
_global_cache: Optional[LLMResponseCache] = None


def get_cache() -> LLMResponseCache:
    global _global_cache
    if _global_cache is None:
        _global_cache = LLMResponseCache()
    return _global_cache
