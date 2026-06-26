"""
优雅关停 + 健康探针增强

- 信号处理（SIGTERM/SIGINT）触发优雅关停
- 请求追踪：等待进行中请求完成（grace_period）
- 健康探针：Liveness + Readiness + Startup 三探针
- 关停钩子：支持注册自定义清理函数
"""
import os
import signal
import time
import threading
import logging
from typing import Callable, Optional, Dict, Any, List
from enum import Enum

logger = logging.getLogger(__name__)


class ShutdownPhase(str, Enum):
    RUNNING = "running"
    DRAINING = "draining"      # 停止接收新请求，等待进行中请求
    TERMINATING = "terminating"  # 强制清理
    STOPPED = "stopped"


class GracefulShutdown:
    """
    优雅关停管理器

    K8s 生命周期:
      1. Pod 收到 SIGTERM
      2. preStop hook 执行
      3. 从 Endpoints 移除（停止新流量）
      4. 等待 grace_period 让进行中请求完成
      5. 关闭进程

    使用:
        manager = GracefulShutdown(grace_period_seconds=30)
        manager.register_hook("redis", close_redis)
        manager.register_hook("db", close_db)
        manager.install_signal_handlers()
    """

    def __init__(
        self,
        grace_period_seconds: float = 30.0,
        drain_check_interval: float = 0.5,
    ):
        self.grace_period_seconds = grace_period_seconds
        self.drain_check_interval = drain_check_interval

        self._phase = ShutdownPhase.RUNNING
        self._active_requests: int = 0
        self._lock = threading.Lock()
        self._hooks: List[Dict[str, Any]] = []
        self._started_at: float = time.time()

    @property
    def phase(self) -> ShutdownPhase:
        return self._phase

    @property
    def is_running(self) -> bool:
        return self._phase == ShutdownPhase.RUNNING

    def register_hook(self, name: str, func: Callable[[], None], priority: int = 100) -> None:
        """注册关停钩子（priority 越小越先执行）"""
        self._hooks.append({"name": name, "func": func, "priority": priority})
        self._hooks.sort(key=lambda h: h["priority"])

    def request_enter(self) -> None:
        """请求进入（用于追踪进行中请求数）"""
        with self._lock:
            self._active_requests += 1

    def request_exit(self) -> None:
        """请求退出"""
        with self._lock:
            self._active_requests = max(0, self._active_requests - 1)

    def install_signal_handlers(self) -> None:
        """安装信号处理器（仅主线程调用）"""
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self._handle_signal)

    def _handle_signal(self, signum: int, frame: Any) -> None:
        sig_name = signal.Signals(signum).name
        logger.info(f"收到信号 {sig_name}，开始优雅关停...")
        self._shutdown()

    def _shutdown(self) -> None:
        if self._phase != ShutdownPhase.RUNNING:
            return

        self._phase = ShutdownPhase.DRAINING
        logger.info(f"进入 DRAINING 阶段，等待 {self.grace_period_seconds}s 让进行中请求完成")

        # 等待进行中请求完成或超时
        deadline = time.time() + self.grace_period_seconds
        while time.time() < deadline:
            with self._lock:
                if self._active_requests <= 0:
                    break
            time.sleep(self.drain_check_interval)

        self._phase = ShutdownPhase.TERMINATING
        logger.info("进入 TERMINATING 阶段，执行关停钩子")

        # 按优先级执行钩子
        for hook in self._hooks:
            try:
                hook["func"]()
                logger.info(f"关停钩子 [{hook['name']}] 执行成功")
            except Exception as e:
                logger.error(f"关停钩子 [{hook['name']}] 执行失败: {e}")

        self._phase = ShutdownPhase.STOPPED
        logger.info("优雅关停完成")

    def get_status(self) -> Dict[str, Any]:
        return {
            "phase": self._phase.value,
            "active_requests": self._active_requests,
            "uptime_seconds": round(time.time() - self._started_at, 1),
            "grace_period_seconds": self.grace_period_seconds,
            "hooks_count": len(self._hooks),
        }


class HealthProbe:
    """
    K8s 三探针实现

    - /health/live    → Liveness:  进程存活？
    - /health/ready   → Readiness: 可以接收流量？
    - /health/startup → Startup:   启动完成？
    """

    def __init__(self, shutdown_manager: Optional[GracefulShutdown] = None):
        self._shutdown = shutdown_manager or GracefulShutdown()
        self._ready: bool = False
        self._started: bool = False
        self._checks: Dict[str, Callable[[], bool]] = {}

    def mark_started(self) -> None:
        self._started = True

    def mark_ready(self, ready: bool = True) -> None:
        self._ready = ready

    def register_check(self, name: str, func: Callable[[], bool]) -> None:
        """注册自定义健康检查"""
        self._checks[name] = func

    def liveness(self) -> Dict[str, Any]:
        """Liveness 探针：进程是否存活"""
        return {
            "status": "ok" if self._shutdown.phase != ShutdownPhase.STOPPED else "dead",
            "phase": self._shutdown.phase.value,
        }

    def readiness(self) -> Dict[str, Any]:
        """Readiness 探针：是否可接收流量"""
        if self._shutdown.phase != ShutdownPhase.RUNNING:
            return {"status": "not_ready", "reason": f"phase={self._shutdown.phase.value}"}

        if not self._ready:
            return {"status": "not_ready", "reason": "not_marked_ready"}

        # 执行自定义检查
        failures = []
        for name, check in self._checks.items():
            try:
                if not check():
                    failures.append(name)
            except Exception as e:
                failures.append(f"{name}:{e}")

        if failures:
            return {"status": "degraded", "failures": failures}

        return {"status": "ok"}

    def startup(self) -> Dict[str, Any]:
        """Startup 探针：启动是否完成"""
        if not self._started:
            return {"status": "starting"}
        return {"status": "ok"}


# 全局单例
_shutdown_manager: Optional[GracefulShutdown] = None
_health_probe: Optional[HealthProbe] = None


def get_shutdown_manager() -> GracefulShutdown:
    global _shutdown_manager
    if _shutdown_manager is None:
        grace = float(os.getenv("GRACE_PERIOD_SECONDS", "30"))
        _shutdown_manager = GracefulShutdown(grace_period_seconds=grace)
    return _shutdown_manager


def get_health_probe() -> HealthProbe:
    global _health_probe
    if _health_probe is None:
        _health_probe = HealthProbe(get_shutdown_manager())
    return _health_probe
