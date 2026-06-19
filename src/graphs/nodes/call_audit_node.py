# -*- coding: utf-8 -*-
"""
调用审计 & 质量监控中间件 - V5.3
- 滑动窗口统计：成功率/平均耗时/P95
- 调用历史：最近1000次
- Prometheus指标格式
"""

import os
import json
import time
import hashlib
import threading
import logging
from collections import deque
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context
from graphs.state import CallAuditInput, CallAuditOutput

logger = logging.getLogger(__name__)


class AuditStore:
    """线程安全的审计存储（单实例）"""
    _instance: Optional["AuditStore"] = None
    _lock = threading.Lock()

    def __init__(self, window_size: int = 100, history_size: int = 1000):
        self._window_size = window_size
        self._history_size = history_size
        self._recent: deque = deque(maxlen=window_size)
        self._history: deque = deque(maxlen=history_size)
        self._total_count = 0
        self._total_success = 0
        self._total_duration = 0.0
        self._lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "AuditStore":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def record(self, audit_record: Dict[str, Any]) -> None:
        with self._lock:
            self._recent.append(audit_record)
            self._history.append(audit_record)
            self._total_count += 1
            if audit_record.get("success", False):
                self._total_success += 1
            self._total_duration += float(audit_record.get("total_duration", 0.0))

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            recent_list = list(self._recent)
        if not recent_list:
            return {
                "total_calls": self._total_count,
                "total_success": self._total_success,
                "window_size": 0,
                "window_success_rate": 1.0,
                "window_avg_duration": 0.0,
                "window_p95_duration": 0.0,
                "all_time_success_rate": 1.0 if self._total_count == 0 else self._total_success / self._total_count,
                "all_time_avg_duration": 0.0 if self._total_count == 0 else self._total_duration / self._total_count,
            }
        success = sum(1 for r in recent_list if r.get("success"))
        durations = sorted([float(r.get("total_duration", 0.0)) for r in recent_list])
        avg = sum(durations) / len(durations)
        p95_idx = int(len(durations) * 0.95)
        p95 = durations[min(p95_idx, len(durations) - 1)]
        return {
            "total_calls": self._total_count,
            "total_success": self._total_success,
            "window_size": len(recent_list),
            "window_success_rate": round(success / len(recent_list), 4),
            "window_avg_duration": round(avg, 4),
            "window_p95_duration": round(p95, 4),
            "all_time_success_rate": round(self._total_success / max(self._total_count, 1), 4),
            "all_time_avg_duration": round(self._total_duration / max(self._total_count, 1), 4),
        }

    def get_recent(self, n: int = 20) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._history)[-n:]

    def to_prometheus(self) -> str:
        """导出Prometheus文本格式"""
        s = self.get_stats()
        lines = [
            "# HELP packcv_ocr_total_calls Total OCR calls",
            f"# TYPE packcv_ocr_total_calls counter",
            f"packcv_ocr_total_calls {s['total_calls']}",
            f"# HELP packcv_ocr_success_rate Window success rate",
            f"# TYPE packcv_ocr_success_rate gauge",
            f"packcv_ocr_success_rate {s['window_success_rate']}",
            f"# HELP packcv_ocr_avg_duration Average duration",
            f"# TYPE packcv_ocr_avg_duration gauge",
            f"packcv_ocr_avg_duration {s['window_avg_duration']}",
            f"# HELP packcv_ocr_p95_duration P95 duration",
            f"# TYPE packcv_ocr_p95_duration gauge",
            f"packcv_ocr_p95_duration {s['window_p95_duration']}",
        ]
        return "\n".join(lines) + "\n"


def call_audit_node(
    state: CallAuditInput,
    config: RunnableConfig,
    runtime: Runtime[Context]
) -> CallAuditOutput:
    """
    title: 调用审计
    desc: 记录每次OCR调用的指标（耗时/成功率/错误），输出Prometheus格式指标和滑动窗口统计
    integrations: 标准logging
    """
    ctx = runtime.context
    audit_store = AuditStore.get_instance()

    # 派生 image_url（兼容多种来源）
    image_url = state.image_url
    if not image_url:
        # 尝试从audit_log_payload中取
        payload = state.audit_log_payload or {}
        image_url = payload.get("image_url", "") or payload.get("image", "")

    image_hash = state.image_hash
    if not image_hash and image_url:
        image_hash = hashlib.md5(image_url.encode("utf-8")).hexdigest()[:16]

    now = time.time()
    start_time = state.start_time or now
    total_duration = max(0.0, now - start_time)

    audit_record = {
        "request_id": state.request_id,
        "caller": state.caller,
        "image_url_hash": hash(image_url) if image_url else 0,
        "image_hash": image_hash,
        "start_time": start_time,
        "end_time": now,
        "total_duration": total_duration,
        "success": state.success,
        "node_metrics": state.node_metrics,
        "error_message": state.error_message,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
    }
    audit_store.record(audit_record)
    stats = audit_store.get_stats()

    audit_id = f"audit_{int(now * 1000)}_{hash(state.request_id) % 10000:04d}"
    logger.info(
        f"[审计] {audit_id} | 耗时{total_duration:.2f}s | "
        f"成功={state.success} | 窗口成功率={stats['window_success_rate']:.2%}"
    )

    return CallAuditOutput(
        audit_id=audit_id,
        audit_log=audit_record,
        total_duration=total_duration,
        success_rate_window=stats["window_success_rate"],
        avg_duration_window=stats["window_avg_duration"]
    )
