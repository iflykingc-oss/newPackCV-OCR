#!/usr/bin/env python3
"""监控指标收集器 - 收集性能、准确率和业务指标"""
import os
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
from collections import defaultdict


class MetricsCollector:
    """监控指标收集器"""
    
    def __init__(self):
        self.metrics: Dict[str, List[Any]] = defaultdict(list)
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = {}
        self.start_time: float = time.time()
    
    def record_latency(self, node_name: str, latency_ms: float) -> None:
        """记录节点延迟"""
        self.metrics[f"latency_{node_name}"].append({
            "timestamp": datetime.utcnow().isoformat(),
            "value": latency_ms,
            "unit": "ms"
        })
    
    def record_confidence(self, node_name: str, confidence: float) -> None:
        """记录置信度"""
        self.metrics[f"confidence_{node_name}"].append({
            "timestamp": datetime.utcnow().isoformat(),
            "value": confidence,
            "unit": "ratio"
        })
    
    def increment_counter(self, metric_name: str, value: int = 1) -> None:
        """增加计数器"""
        self.counters[metric_name] += value
    
    def set_gauge(self, metric_name: str, value: float) -> None:
        """设置仪表值"""
        self.gauges[metric_name] = value
    
    def record_scenario_detection(self, scenario: str, confidence: float) -> None:
        """记录场景检测结果"""
        self.increment_counter(f"scenario_{scenario}")
        self.record_confidence("scenario_detector", confidence)
    
    def record_extraction_result(self, fields_extracted: int, fields_attempted: int) -> None:
        """记录提取结果"""
        self.increment_counter("extractions_total")
        self.increment_counter("fields_extracted", fields_extracted)
        self.increment_counter("fields_attempted", fields_attempted)
        
        # 计算提取成功率
        if fields_attempted > 0:
            success_rate: float = fields_extracted / fields_attempted
            self.set_gauge("extraction_success_rate", success_rate)
    
    def record_batch_processing(self, total: int, success: int, failed: int) -> None:
        """记录批量处理结果"""
        self.increment_counter("batch_total", total)
        self.increment_counter("batch_success", success)
        self.increment_counter("batch_failed", failed)
        
        if total > 0:
            success_rate: float = success / total
            self.set_gauge("batch_success_rate", success_rate)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """获取指标摘要"""
        elapsed_time: float = time.time() - self.start_time
        
        # 计算平均延迟
        avg_latencies: Dict[str, float] = {}
        for metric_name, values in self.metrics.items():
            if metric_name.startswith("latency_"):
                if values:
                    avg_value: float = sum(v["value"] for v in values) / len(values)
                    avg_latencies[metric_name.replace("latency_", "")] = round(avg_value, 2)
        
        # 计算平均置信度
        avg_confidences: Dict[str, float] = {}
        for metric_name, values in self.metrics.items():
            if metric_name.startswith("confidence_"):
                if values:
                    avg_value: float = sum(v["value"] for v in values) / len(values)
                    avg_confidences[metric_name.replace("confidence_", "")] = round(avg_value, 3)
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "elapsed_time_seconds": round(elapsed_time, 2),
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "avg_latencies_ms": avg_latencies,
            "avg_confidences": avg_confidences,
            "scenario_distribution": {
                k.replace("scenario_", ""): v 
                for k, v in self.counters.items() 
                if k.startswith("scenario_")
            },
        }
    
    def export_to_json(self) -> str:
        """导出为JSON字符串"""
        return json.dumps(self.get_metrics_summary(), ensure_ascii=False)


# 全局指标收集器
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """获取全局指标收集器"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def reset_metrics_collector() -> None:
    """重置指标收集器"""
    global _metrics_collector
    _metrics_collector = MetricsCollector()


class NodeMetrics:
    """节点级别的指标记录器"""
    
    def __init__(self, node_name: str):
        self.node_name: str = node_name
        self.collector: MetricsCollector = get_metrics_collector()
        self.start_time: float = 0.0
    
    def start(self) -> None:
        """开始记录"""
        self.start_time = time.time()
    
    def end(self, confidence: Optional[float] = None, **extra_metrics) -> None:
        """结束记录"""
        latency: float = (time.time() - self.start_time) * 1000
        self.collector.record_latency(self.node_name, latency)
        
        if confidence is not None:
            self.collector.record_confidence(self.node_name, confidence)
        
        for metric_name, value in extra_metrics.items():
            if isinstance(value, int):
                self.collector.increment_counter(f"{self.node_name}_{metric_name}", value)
            elif isinstance(value, float):
                self.collector.set_gauge(f"{self.node_name}_{metric_name}", value)