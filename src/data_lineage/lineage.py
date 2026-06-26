#!/usr/bin/env python3
"""数据血缘追踪模块
功能:
- 每条提取记录可溯源到原始文件
- 记录 LLM 模型 + 参数
- 记录处理路径 (节点链)
- 支持血缘查询和可视化
"""
import os
import json
import logging
import uuid
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class LineageNode(BaseModel):
    """血缘节点"""
    node_id: str = Field(..., description="节点ID")
    node_type: str = Field(..., description="节点类型 (source/process/model/output)")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LineageEdge(BaseModel):
    """血缘边"""
    from_node: str = Field(..., description="源节点ID")
    to_node: str = Field(..., description="目标节点ID")
    edge_type: str = Field(default="data_flow", description="边类型")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DataLineage(BaseModel):
    """数据血缘记录"""
    lineage_id: str = Field(..., description="血缘ID")
    run_id: str = Field(..., description="工作流运行ID")
    tenant_id: str = Field(default="unknown", description="租户ID")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    
    # 数据来源
    source_file_url: str = Field(default="", description="原始文件URL")
    source_file_type: str = Field(default="", description="文件类型")
    source_file_hash: str = Field(default="", description="文件哈希")
    
    # 处理路径
    nodes: List[LineageNode] = Field(default_factory=list, description="处理节点")
    edges: List[LineageEdge] = Field(default_factory=list, description="流转边")
    
    # 模型信息
    model_provider: str = Field(default="", description="LLM Provider")
    model_name: str = Field(default="", description="模型名称")
    model_params: Dict[str, Any] = Field(default_factory=dict, description="模型参数")
    
    # 输出信息
    output_schema: str = Field(default="", description="输出 Schema")
    output_hash: str = Field(default="", description="输出哈希")
    
    # 质量信息
    confidence: float = Field(default=0.0, description="置信度")
    validation_passed: bool = Field(default=False, description="校验通过")


class LineageStore:
    """血缘存储"""
    
    _store: Dict[str, DataLineage] = {}
    _tenant_index: Dict[str, List[str]] = {}  # tenant_id -> lineage_ids
    _run_index: Dict[str, str] = {}  # run_id -> lineage_id
    
    @classmethod
    def save(cls, lineage: DataLineage) -> str:
        """保存血缘记录"""
        cls._store[lineage.lineage_id] = lineage
        
        # 索引
        tenant = lineage.tenant_id
        if tenant not in cls._tenant_index:
            cls._tenant_index[tenant] = []
        cls._tenant_index[tenant].append(lineage.lineage_id)
        
        cls._run_index[lineage.run_id] = lineage.lineage_id
        
        logger.debug(f"Lineage saved: {lineage.lineage_id}")
        return lineage.lineage_id
    
    @classmethod
    def get(cls, lineage_id: str) -> Optional[DataLineage]:
        """获取血缘记录"""
        return cls._store.get(lineage_id)
    
    @classmethod
    def get_by_run(cls, run_id: str) -> Optional[DataLineage]:
        """按运行ID查询"""
        lineage_id = cls._run_index.get(run_id)
        if lineage_id:
            return cls._store.get(lineage_id)
        return None
    
    @classmethod
    def list_by_tenant(cls, tenant_id: str, limit: int = 100) -> List[DataLineage]:
        """按租户查询"""
        ids = cls._tenant_index.get(tenant_id, [])[:limit]
        return [cls._store[id_] for id_ in ids if id_ in cls._store]
    
    @classmethod
    def search(cls, query: Dict[str, Any], limit: int = 50) -> List[DataLineage]:
        """搜索血缘"""
        results = []
        for lineage in cls._store.values():
            match = True
            for key, value in query.items():
                if hasattr(lineage, key):
                    if getattr(lineage, key) != value:
                        match = False
                        break
            if match:
                results.append(lineage)
                if len(results) >= limit:
                    break
        return results
    
    @classmethod
    def stats(cls) -> Dict[str, Any]:
        """统计信息"""
        return {
            "total_lineages": len(cls._store),
            "total_tenants": len(cls._tenant_index),
            "total_runs": len(cls._run_index),
        }
    
    @classmethod
    def clear(cls, tenant_id: Optional[str] = None):
        """清理"""
        if tenant_id:
            ids = cls._tenant_index.pop(tenant_id, [])
            for id_ in ids:
                lineage = cls._store.pop(id_, None)
                if lineage:
                    cls._run_index.pop(lineage.run_id, None)
        else:
            cls._store.clear()
            cls._tenant_index.clear()
            cls._run_index.clear()


class LineageBuilder:
    """血缘构建器"""
    
    def __init__(self, run_id: str, tenant_id: str = "unknown"):
        self.lineage_id = f"lin_{uuid.uuid4().hex[:12]}"
        self.run_id = run_id
        self.tenant_id = tenant_id
        self.nodes: List[LineageNode] = []
        self.edges: List[LineageEdge] = []
        self._last_node_id: Optional[str] = None
    
    def add_source(
        self,
        file_url: str,
        file_type: str,
        file_hash: Optional[str] = None
    ) -> str:
        """添加数据源节点"""
        node_id = f"src_{uuid.uuid4().hex[:8]}"
        self.nodes.append(LineageNode(
            node_id=node_id,
            node_type="source",
            metadata={
                "file_url": file_url,
                "file_type": file_type,
                "file_hash": file_hash or self._compute_hash(file_url)
            }
        ))
        self._last_node_id = node_id
        return node_id
    
    def add_process(
        self,
        process_name: str,
        duration_ms: float = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """添加处理节点"""
        node_id = f"proc_{uuid.uuid4().hex[:8]}"
        self.nodes.append(LineageNode(
            node_id=node_id,
            node_type="process",
            metadata={
                "process_name": process_name,
                "duration_ms": duration_ms,
                **(metadata or {})
            }
        ))
        
        # 添加边
        if self._last_node_id:
            self.edges.append(LineageEdge(
                from_node=self._last_node_id,
                to_node=node_id
            ))
        
        self._last_node_id = node_id
        return node_id
    
    def add_model(
        self,
        provider: str,
        model: str,
        params: Optional[Dict[str, Any]] = None,
        tokens_in: int = 0,
        tokens_out: int = 0
    ) -> str:
        """添加模型节点"""
        node_id = f"model_{uuid.uuid4().hex[:8]}"
        self.nodes.append(LineageNode(
            node_id=node_id,
            node_type="model",
            metadata={
                "provider": provider,
                "model": model,
                "params": params or {},
                "tokens_in": tokens_in,
                "tokens_out": tokens_out
            }
        ))
        
        if self._last_node_id:
            self.edges.append(LineageEdge(
                from_node=self._last_node_id,
                to_node=node_id
            ))
        
        self._last_node_id = node_id
        return node_id
    
    def add_output(
        self,
        schema: str,
        output_hash: str,
        confidence: float = 0.0
    ) -> str:
        """添加输出节点"""
        node_id = f"out_{uuid.uuid4().hex[:8]}"
        self.nodes.append(LineageNode(
            node_id=node_id,
            node_type="output",
            metadata={
                "schema": schema,
                "output_hash": output_hash,
                "confidence": confidence
            }
        ))
        
        if self._last_node_id:
            self.edges.append(LineageEdge(
                from_node=self._last_node_id,
                to_node=node_id
            ))
        
        return node_id
    
    def build(
        self,
        source_file_url: str = "",
        source_file_type: str = "",
        source_file_hash: str = "",
        model_provider: str = "",
        model_name: str = "",
        model_params: Dict[str, Any] = None,
        output_schema: str = "",
        output_hash: str = "",
        confidence: float = 0.0
    ) -> DataLineage:
        """构建完整血缘"""
        return DataLineage(
            lineage_id=self.lineage_id,
            run_id=self.run_id,
            tenant_id=self.tenant_id,
            source_file_url=source_file_url,
            source_file_type=source_file_type,
            source_file_hash=source_file_hash,
            nodes=self.nodes,
            edges=self.edges,
            model_provider=model_provider,
            model_name=model_name,
            model_params=model_params or {},
            output_schema=output_schema,
            output_hash=output_hash,
            confidence=confidence
        )
    
    @staticmethod
    def _compute_hash(data: str) -> str:
        """计算哈希"""
        import hashlib
        return hashlib.md5(data.encode()).hexdigest()[:12]


def create_lineage(run_id: str, tenant_id: str = "unknown") -> LineageBuilder:
    """创建血缘构建器"""
    return LineageBuilder(run_id, tenant_id)


def get_lineage(lineage_id: str) -> Optional[DataLineage]:
    """获取血缘"""
    return LineageStore.get(lineage_id)


def get_lineage_by_run(run_id: str) -> Optional[DataLineage]:
    """按运行ID获取血缘"""
    return LineageStore.get_by_run(run_id)


def list_lineages_by_tenant(tenant_id: str, limit: int = 100) -> List[DataLineage]:
    """按租户列出血缘"""
    return LineageStore.list_by_tenant(tenant_id, limit)


def lineage_stats() -> Dict[str, Any]:
    """血缘统计"""
    return LineageStore.stats()