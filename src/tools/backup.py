"""
数据备份恢复工具

支持:
  - Redis 全量备份（SCAN + DUMP）
  - 租户数据导出（JSON）
  - 配置快照
  - 恢复验证
"""
import os
import json
import time
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)


class DataBackup:
    """Redis 数据备份管理器"""

    def __init__(self, output_dir: str = "/tmp/packcv-backups"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _get_redis(self):
        from utils.redis_client import redis_client
        return redis_client.client

    def backup_all(self, label: Optional[str] = None) -> str:
        """全量备份 Redis 数据到 JSON"""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        label = label or "full"
        filename = f"backup_{label}_{ts}.json"
        filepath = os.path.join(self.output_dir, filename)

        r = self._get_redis()
        data: Dict[str, Any] = {
            "metadata": {
                "timestamp": ts,
                "label": label,
                "version": "7.0.0",
            },
            "keys": {},
        }

        cursor = 0
        total = 0
        while True:
            cursor, keys = r.scan(cursor, count=500)
            for key in keys:
                key_str = key.decode() if isinstance(key, bytes) else key
                key_type = r.type(key_str)
                if isinstance(key_type, bytes):
                    key_type = key_type.decode()

                if key_type == "string":
                    val = r.get(key_str)
                    data["keys"][key_str] = {
                        "type": "string",
                        "value": val.decode() if isinstance(val, bytes) else val,
                    }
                elif key_type == "hash":
                    val = r.hgetall(key_str)
                    data["keys"][key_str] = {
                        "type": "hash",
                        "value": {k.decode() if isinstance(k, bytes) else k: v.decode() if isinstance(v, bytes) else v for k, v in val.items()},
                    }
                elif key_type == "set":
                    val = r.smembers(key_str)
                    data["keys"][key_str] = {
                        "type": "set",
                        "value": list(v.decode() if isinstance(v, bytes) else v for v in val),
                    }
                elif key_type == "list":
                    val = r.lrange(key_str, 0, -1)
                    data["keys"][key_str] = {
                        "type": "list",
                        "value": [v.decode() if isinstance(v, bytes) else v for v in val],
                    }
                elif key_type == "zset":
                    val = r.zrange(key_str, 0, -1, withscores=True)
                    data["keys"][key_str] = {
                        "type": "zset",
                        "value": [(m.decode() if isinstance(m, bytes) else m, s) for m, s in val],
                    }
                total += 1

            if cursor == 0:
                break

        data["metadata"]["total_keys"] = total

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"备份完成: {filepath} ({total} keys)")
        return filepath

    def backup_tenant(self, tenant_id: str) -> str:
        """导出指定租户数据"""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tenant_{tenant_id}_{ts}.json"
        filepath = os.path.join(self.output_dir, filename)

        r = self._get_redis()
        data: Dict[str, Any] = {
            "metadata": {"tenant_id": tenant_id, "timestamp": ts},
            "tenant_data": {},
        }

        # 搜索该租户相关的 key
        for prefix in [f"tenants:{tenant_id}", f"billing:{tenant_id}",
                       f"api_keys:{tenant_id}", f"rate_limit:{tenant_id}",
                       f"audit:{tenant_id}", f"webhook:{tenant_id}"]:
            cursor = 0
            while True:
                cursor, keys = r.scan(cursor, match=f"{prefix}*", count=200)
                for key in keys:
                    key_str = key.decode() if isinstance(key, bytes) else key
                    key_type = r.type(key_str)
                    if isinstance(key_type, bytes):
                        key_type = key_type.decode()
                    if key_type == "hash":
                        val = r.hgetall(key_str)
                        data["tenant_data"][key_str] = {
                            k.decode() if isinstance(k, bytes) else k: v.decode() if isinstance(v, bytes) else v
                            for k, v in val.items()
                        }
                    elif key_type == "string":
                        val = r.get(key_str)
                        data["tenant_data"][key_str] = val.decode() if isinstance(val, bytes) else val
                if cursor == 0:
                    break

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"租户备份完成: {filepath}")
        return filepath

    def restore(self, filepath: str, dry_run: bool = True) -> Dict[str, Any]:
        """从备份恢复数据"""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        r = self._get_redis()
        restored = 0
        skipped = 0
        errors = 0

        for key_str, value_data in data.get("keys", data.get("tenant_data", {})).items():
            try:
                if isinstance(value_data, dict) and "type" in value_data:
                    key_type = value_data["type"]
                    value = value_data["value"]
                else:
                    key_type = "string"
                    value = value_data

                if dry_run:
                    skipped += 1
                    continue

                if key_type == "string":
                    r.set(key_str, value)
                elif key_type == "hash":
                    r.hset(key_str, mapping=value)
                elif key_type == "set":
                    for v in value:
                        r.sadd(key_str, v)
                elif key_type == "list":
                    for v in value:
                        r.rpush(key_str, v)
                restored += 1

            except Exception as e:
                logger.error(f"恢复 {key_str} 失败: {e}")
                errors += 1

        result = {
            "file": filepath,
            "dry_run": dry_run,
            "restored": restored,
            "skipped": skipped,
            "errors": errors,
        }
        logger.info(f"恢复完成: {result}")
        return result

    def list_backups(self) -> List[Dict[str, Any]]:
        """列出所有备份"""
        backups = []
        for f in sorted(os.listdir(self.output_dir)):
            if f.endswith(".json"):
                filepath = os.path.join(self.output_dir, f)
                stat = os.stat(filepath)
                backups.append({
                    "filename": f,
                    "size_kb": round(stat.st_size / 1024, 1),
                    "created": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                })
        return backups
