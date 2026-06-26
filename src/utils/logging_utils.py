#!/usr/bin/env python3
"""结构化日志工具 - 统一日志格式，支持追踪ID和性能监控"""
import os
import json
import logging
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Optional

# 创建自定义日志格式
class StructuredFormatter(logging.Formatter):
    """结构化日志格式器，输出JSON格式日志"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # 添加额外字段
        if hasattr(record, 'trace_id'):
            log_data["trace_id"] = record.trace_id
        if hasattr(record, 'node_name'):
            log_data["node_name"] = record.node_name
        if hasattr(record, 'processing_time_ms'):
            log_data["processing_time_ms"] = record.processing_time_ms
        if hasattr(record, 'confidence'):
            log_data["confidence"] = record.confidence
        if hasattr(record, 'scenario'):
            log_data["scenario"] = record.scenario
        if hasattr(record, 'status'):
            log_data["status"] = record.status
        
        # 添加异常信息
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data, ensure_ascii=False)


class StructuredLogger:
    """结构化日志记录器"""
    
    def __init__(self, name: str, trace_id: Optional[str] = None):
        self.logger = logging.getLogger(name)
        self.trace_id: str = trace_id or self._generate_trace_id()
        
        # 配置日志处理器
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = StructuredFormatter()
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def _generate_trace_id(self) -> str:
        """生成唯一的追踪ID"""
        return f"trace_{uuid.uuid4().hex[:16]}"
    
    def _add_extra(self, extra: Dict[str, Any]) -> Dict[str, Any]:
        """添加trace_id到extra字段"""
        extra["trace_id"] = self.trace_id
        return extra
    
    def info(self, message: str, **kwargs) -> None:
        """记录INFO级别日志"""
        extra: Dict[str, Any] = self._add_extra(kwargs)
        self.logger.info(message, extra=extra)
    
    def warning(self, message: str, **kwargs) -> None:
        """记录WARNING级别日志"""
        extra: Dict[str, Any] = self._add_extra(kwargs)
        self.logger.warning(message, extra=extra)
    
    def error(self, message: str, **kwargs) -> None:
        """记录ERROR级别日志"""
        extra: Dict[str, Any] = self._add_extra(kwargs)
        self.logger.error(message, extra=extra)
    
    def debug(self, message: str, **kwargs) -> None:
        """记录DEBUG级别日志"""
        extra: Dict[str, Any] = self._add_extra(kwargs)
        self.logger.debug(message, extra=extra)


class PerformanceTracker:
    """性能追踪器，记录节点执行时间和性能指标"""
    
    def __init__(self, trace_id: Optional[str] = None):
        self.trace_id: str = trace_id or f"perf_{uuid.uuid4().hex[:12]}"
        self.start_time: float = time.time()
        self.metrics: Dict[str, Dict[str, Any]] = {}
    
    def start_node(self, node_name: str) -> float:
        """开始追踪节点执行"""
        node_start: float = time.time()
        self.metrics[node_name] = {
            "start_time": node_start,
            "status": "running"
        }
        return node_start
    
    def end_node(self, node_name: str, confidence: Optional[float] = None, status: str = "success") -> float:
        """结束节点追踪"""
        node_end: float = time.time()
        if node_name in self.metrics:
            node_data: Dict[str, Any] = self.metrics[node_name]
            node_data["end_time"] = node_end
            node_data["processing_time_ms"] = (node_end - node_data["start_time"]) * 1000
            node_data["status"] = status
            if confidence is not None:
                node_data["confidence"] = confidence
            return node_data["processing_time_ms"]
        return 0.0
    
    def get_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        total_time: float = (time.time() - self.start_time) * 1000
        
        node_times: Dict[str, float] = {}
        total_confidence: float = 0.0
        confidence_count: int = 0
        
        for node_name, data in self.metrics.items():
            if "processing_time_ms" in data:
                node_times[node_name] = data["processing_time_ms"]
            if "confidence" in data:
                total_confidence += data["confidence"]
                confidence_count += 1
        
        avg_confidence: float = total_confidence / confidence_count if confidence_count > 0 else 0.0
        
        return {
            "trace_id": self.trace_id,
            "total_time_ms": round(total_time, 2),
            "node_times": node_times,
            "avg_confidence": round(avg_confidence, 3),
            "nodes_count": len(self.metrics),
            "success_count": sum(1 for d in self.metrics.values() if d.get("status") == "success"),
            "failed_count": sum(1 for d in self.metrics.values() if d.get("status") == "failed"),
        }


# 创建全局日志记录器工厂
def get_logger(name: str, trace_id: Optional[str] = None) -> StructuredLogger:
    """获取结构化日志记录器"""
    return StructuredLogger(name, trace_id)


def get_performance_tracker(trace_id: Optional[str] = None) -> PerformanceTracker:
    """获取性能追踪器"""
    return PerformanceTracker(trace_id)