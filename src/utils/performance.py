import os
import logging
from pathlib import Path
from typing import Dict, Any, List
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing

logger = logging.getLogger(__name__)


class PerformanceOptimizer:
    """性能优化器"""

    def __init__(self):
        self.cpu_count = multiprocessing.cpu_count()
        self._process_pool: ProcessPoolExecutor = None
        self._thread_pool: ThreadPoolExecutor = None

    @property
    def process_pool(self) -> ProcessPoolExecutor:
        """获取进程池（计算密集型任务）"""
        if self._process_pool is None:
            max_workers = max(1, self.cpu_count - 1)
            self._process_pool = ProcessPoolExecutor(max_workers=max_workers)
            logger.info(f"进程池初始化: {max_workers} workers")
        return self._process_pool

    @property
    def thread_pool(self) -> ThreadPoolExecutor:
        """获取线程池（IO密集型任务）"""
        if self._thread_pool is None:
            max_workers = self.cpu_count * 2
            self._thread_pool = ThreadPoolExecutor(max_workers=max_workers)
            logger.info(f"线程池初始化: {max_workers} workers")
        return self._thread_pool

    def submit_process(self, func, *args, **kwargs):
        """提交计算密集型任务到进程池"""
        return self.process_pool.submit(func, *args, **kwargs)

    def submit_thread(self, func, *args, **kwargs):
        """提交IO密集型任务到线程池"""
        return self.thread_pool.submit(func, *args, **kwargs)

    def map_process(self, func, iterable, chunksize: int = 1):
        """进程池批量处理"""
        return self.process_pool.map(func, iterable, chunksize=chunksize)

    def map_thread(self, func, iterable):
        """线程池批量处理"""
        return self.thread_pool.map(func, iterable)

    def parallel_ocr(
        self,
        images: List[bytes],
        ocr_func,
        max_workers: int = None
    ) -> List[Dict[str, Any]]:
        """
        并行OCR识别

        Args:
            images: 图片列表
            ocr_func: OCR识别函数
            max_workers: 最大并行数

        Returns:
            识别结果列表
        """
        results = []

        with ThreadPoolExecutor(max_workers=max_workers or self.cpu_count) as executor:
            futures = {
                executor.submit(ocr_func, img): i
                for i, img in enumerate(images)
            }

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    results.append({"index": idx, "result": result, "error": None})
                except Exception as e:
                    logger.error(f"OCR任务 {idx} 失败: {e}")
                    results.append({"index": idx, "result": None, "error": str(e)})

        # 按原始顺序排序
        results.sort(key=lambda x: x["index"])
        return results

    def parallel_detection(
        self,
        images: List[bytes],
        detect_func,
        max_workers: int = None
    ) -> List[Dict[str, Any]]:
        """
        并行目标检测

        Args:
            images: 图片列表
            detect_func: 检测函数
            max_workers: 最大并行数

        Returns:
            检测结果列表
        """
        results = []

        with ProcessPoolExecutor(max_workers=max_workers or self.cpu_count) as executor:
            futures = {
                executor.submit(detect_func, img): i
                for i, img in enumerate(images)
            }

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    results.append({"index": idx, "result": result, "error": None})
                except Exception as e:
                    logger.error(f"检测任务 {idx} 失败: {e}")
                    results.append({"index": idx, "result": None, "error": str(e)})

        results.sort(key=lambda x: x["index"])
        return results

    def shutdown(self):
        """关闭线程池和进程池"""
        if self._process_pool:
            self._process_pool.shutdown(wait=True)
            self._process_pool = None
        if self._thread_pool:
            self._thread_pool.shutdown(wait=True)
            self._thread_pool = None


# 全局优化器实例
_optimizer: PerformanceOptimizer = None


def get_optimizer() -> PerformanceOptimizer:
    """获取性能优化器单例"""
    global _optimizer
    if _optimizer is None:
        _optimizer = PerformanceOptimizer()
    return _optimizer


# ============ 任务队列 ============

class TaskQueue:
    """任务队列管理器"""

    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/1")
        self._queue = None
        self._init_queue()

    def _init_queue(self):
        """初始化Redis队列"""
        try:
            from rq import Queue
            from redis import Redis

            redis_conn = Redis.from_url(self.redis_url)
            self._queue = Queue('packcv', connection=redis_conn)
            logger.info("任务队列初始化成功")
        except ImportError:
            logger.warning("rq未安装，使用内存队列")
            self._queue = None
        except Exception as e:
            logger.warning(f"任务队列初始化失败: {e}")
            self._queue = None

    def enqueue(self, func, *args, **kwargs):
        """入队任务"""
        if self._queue:
            return self._queue.enqueue(func, *args, **kwargs)
        else:
            # 同步执行
            return func(*args, **kwargs)

    def enqueue_batch(self, jobs: List[tuple]):
        """批量入队"""
        if self._queue:
            return self._queue.enqueue_batch([
                self._queue.prepare_data(func, args, kwargs)
                for func, args, kwargs in jobs
            ])
        else:
            return [func(*args, **kwargs) for func, args, kwargs in jobs]

    def get_status(self, job_id: str) -> Dict[str, Any]:
        """获取任务状态"""
        if self._queue:
            from rq.job import Job
            job = Job.fetch(job_id, connection=self._queue.connection)
            return {
                "id": job_id,
                "status": job.get_status(),
                "result": job.result if job.is_finished else None,
                "error": job.exc_info if job.is_failed else None
            }
        return {"status": "unknown"}

    def cancel(self, job_id: str) -> bool:
        """取消任务"""
        if self._queue:
            from rq.job import Job
            try:
                job = Job.fetch(job_id, connection=self._queue.connection)
                job.cancel()
                return True
            except Exception as e:
                logger.error(f"取消任务失败: {e}")
                return False
        return False


# ============ 资源监控 ============

class ResourceMonitor:
    """资源监控器"""

    def __init__(self):
        self._metrics: Dict[str, List[float]] = {}

    def record_cpu(self, value: float):
        """记录CPU使用率"""
        self._metrics.setdefault("cpu", []).append(value)
        if len(self._metrics["cpu"]) > 1000:
            self._metrics["cpu"].pop(0)

    def record_memory(self, value: float):
        """记录内存使用率"""
        self._metrics.setdefault("memory", []).append(value)
        if len(self._metrics["memory"]) > 1000:
            self._metrics["memory"].pop(0)

    def record_latency(self, operation: str, value: float):
        """记录延迟"""
        self._metrics.setdefault(f"latency_{operation}", []).append(value)
        if len(self._metrics[f"latency_{operation}"]) > 1000:
            self._metrics[f"latency_{operation}"].pop(0)

    def get_metrics(self) -> Dict[str, Any]:
        """获取当前指标"""
        import psutil

        return {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_available_mb": psutil.virtual_memory().available / (1024 * 1024),
            "disk_percent": psutil.disk_usage('/').percent,
            "latencies": {
                key.replace("latency_", ""): {
                    "avg": sum(vals) / len(vals) if vals else 0,
                    "max": max(vals) if vals else 0,
                    "min": min(vals) if vals else 0,
                    "p95": sorted(vals)[int(len(vals) * 0.95)] if vals and len(vals) > 1 else (vals[0] if vals else 0)
                }
                for key, vals in self._metrics.items()
                if key.startswith("latency_")
            }
        }

    def get_health_status(self) -> str:
        """获取健康状态"""
        metrics = self.get_metrics()

        if metrics["memory_percent"] > 90:
            return "critical"
        elif metrics["memory_percent"] > 75:
            return "warning"
        elif metrics["cpu_percent"] > 90:
            return "critical"
        elif metrics["cpu_percent"] > 75:
            return "warning"
        return "healthy"


# 全局监控器
_monitor: ResourceMonitor = None


def get_monitor() -> ResourceMonitor:
    """获取资源监控器单例"""
    global _monitor
    if _monitor is None:
        _monitor = ResourceMonitor()
    return _monitor
