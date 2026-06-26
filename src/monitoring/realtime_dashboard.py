"""实时仪表盘 + 限流可视化

功能：
- 限流状态实时查询（RPM/TPM/并发）
- 租户级用量热力图数据
- P50/P95/P99 延迟分位
- 端点 TopN 排行
- Prometheus 兼容 metrics
"""
import time
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class LatencySample:
    """延迟采样"""
    timestamp: float
    duration_ms: float
    endpoint: str
    status_code: int


@dataclass
class RateLimitSnapshot:
    """限流快照"""
    tenant_id: str
    rpm_current: int = 0
    rpm_limit: int = 0
    tpm_current: int = 0
    tpm_limit: int = 0
    concurrent_current: int = 0
    concurrent_limit: int = 0
    blocked_count: int = 0
    timestamp: float = 0.0


class RealtimeDashboard:
    """实时仪表盘数据引擎

    特性：
    - 环形缓冲区保存最近 N 条延迟采样
    - 按端点聚合 P50/P95/P99
    - 按租户聚合限流状态
    - TopN 热点端点
    """

    def __init__(self, max_samples: int = 10000) -> None:
        self._max_samples: int = max_samples
        self._samples: List[LatencySample] = []
        self._lock: threading.Lock = threading.Lock()
        self._rate_limit_snapshots: Dict[str, RateLimitSnapshot] = {}
        self._endpoint_counts: Dict[str, int] = defaultdict(int)
        self._error_counts: Dict[str, int] = defaultdict(int)
        self._total_requests: int = 0
        self._start_time: float = time.time()

    def record_latency(
        self, endpoint: str, duration_ms: float, status_code: int = 200
    ) -> None:
        """记录一次请求延迟"""
        with self._lock:
            self._samples.append(LatencySample(
                timestamp=time.time(),
                duration_ms=duration_ms,
                endpoint=endpoint,
                status_code=status_code,
            ))
            if len(self._samples) > self._max_samples:
                self._samples = self._samples[-self._max_samples:]
            self._endpoint_counts[endpoint] += 1
            self._total_requests += 1
            if status_code >= 400:
                self._error_counts[endpoint] += 1

    def update_rate_limit(self, snapshot: RateLimitSnapshot) -> None:
        """更新租户限流状态"""
        snapshot.timestamp = time.time()
        self._rate_limit_snapshots[snapshot.tenant_id] = snapshot

    def get_latency_percentiles(self, endpoint: Optional[str] = None) -> Dict[str, Any]:
        """计算延迟分位"""
        with self._lock:
            samples = self._samples
            if endpoint:
                samples = [s for s in samples if s.endpoint == endpoint]
            if not samples:
                return {"p50": 0, "p95": 0, "p99": 0, "count": 0}
            durations = sorted(s.duration_ms for s in samples)
            n = len(durations)
            return {
                "p50": durations[int(n * 0.50)],
                "p95": durations[int(n * 0.95)],
                "p99": durations[min(int(n * 0.99), n - 1)],
                "avg": sum(durations) / n,
                "min": durations[0],
                "max": durations[-1],
                "count": n,
            }

    def get_top_endpoints(self, n: int = 10) -> List[Dict[str, Any]]:
        """TopN 热点端点"""
        sorted_eps = sorted(self._endpoint_counts.items(), key=lambda x: -x[1])
        result = []
        for ep, count in sorted_eps[:n]:
            errors = self._error_counts.get(ep, 0)
            pct = self.get_latency_percentiles(ep)
            result.append({
                "endpoint": ep,
                "requests": count,
                "errors": errors,
                "error_rate": round(errors / max(count, 1) * 100, 2),
                "p50_ms": round(pct["p50"], 2),
                "p95_ms": round(pct["p95"], 2),
                "p99_ms": round(pct["p99"], 2),
            })
        return result

    def get_rate_limit_status(self) -> List[Dict[str, Any]]:
        """所有租户限流状态"""
        result = []
        for tid, snap in self._rate_limit_snapshots.items():
            rpm_pct = round(snap.rpm_current / max(snap.rpm_limit, 1) * 100, 1)
            tpm_pct = round(snap.tpm_current / max(snap.tpm_limit, 1) * 100, 1)
            result.append({
                "tenant_id": tid,
                "rpm": {"current": snap.rpm_current, "limit": snap.rpm_limit, "pct": rpm_pct},
                "tpm": {"current": snap.tpm_current, "limit": snap.tpm_limit, "pct": tpm_pct},
                "concurrent": {"current": snap.concurrent_current, "limit": snap.concurrent_limit},
                "blocked": snap.blocked_count,
            })
        return result

    def get_overview(self) -> Dict[str, Any]:
        """仪表盘概览"""
        uptime = time.time() - self._start_time
        rps = self._total_requests / max(uptime, 1)
        return {
            "uptime_seconds": round(uptime, 1),
            "total_requests": self._total_requests,
            "requests_per_second": round(rps, 2),
            "unique_endpoints": len(self._endpoint_counts),
            "total_errors": sum(self._error_counts.values()),
            "error_rate": round(
                sum(self._error_counts.values()) / max(self._total_requests, 1) * 100, 2
            ),
        }

    def get_heatmap_data(self, last_n_minutes: int = 60) -> List[Dict[str, Any]]:
        """生成热力图数据（按分钟聚合）"""
        cutoff = time.time() - last_n_minutes * 60
        with self._lock:
            recent = [s for s in self._samples if s.timestamp > cutoff]
        # 按分钟+端点聚合
        buckets: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for s in recent:
            minute_key = time.strftime("%H:%M", time.localtime(s.timestamp))
            buckets[minute_key][s.endpoint] += 1
        result = []
        for minute, eps in sorted(buckets.items()):
            result.append({"minute": minute, "endpoints": dict(eps), "total": sum(eps.values())})
        return result

    def reset(self) -> None:
        """重置所有数据"""
        with self._lock:
            self._samples.clear()
            self._endpoint_counts.clear()
            self._error_counts.clear()
            self._rate_limit_snapshots.clear()
            self._total_requests = 0
            self._start_time = time.time()


# 单例
dashboard = RealtimeDashboard()
