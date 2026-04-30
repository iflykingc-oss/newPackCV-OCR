# -*- coding: utf-8 -*-
"""
基础工作流抽象类
标准化工作流设计模式
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Type
from dataclasses import dataclass, field

from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


@dataclass
class WorkflowConfig:
    """工作流配置"""
    name: str
    description: str
    timeout: int = 300  # 超时时间（秒）
    max_retries: int = 3  # 最大重试次数
    enable_cache: bool = True  # 是否启用缓存
    enable_fallback: bool = True  # 是否启用降级


@dataclass
class WorkflowMetadata:
    """工作流元数据"""
    workflow_id: str
    workflow_name: str
    start_time: float = 0
    end_time: float = 0
    status: str = "pending"  # pending/running/completed/failed
    nodes_executed: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        """执行时长（秒）"""
        if self.end_time and self.start_time:
            return self.end_time - self.start_time
        return 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'workflow_id': self.workflow_id,
            'workflow_name': self.workflow_name,
            'status': self.status,
            'duration': f"{self.duration:.2f}s",
            'nodes_executed': self.nodes_executed,
            'errors': self.errors,
            'warnings': self.warnings
        }


class BaseWorkflow(ABC):
    """
    基础工作流抽象类
    定义工作流的通用接口和生命周期
    """

    def __init__(self, config: Optional[WorkflowConfig] = None):
        self.config = config or WorkflowConfig(
            name=self.__class__.__name__,
            description="工作流"
        )
        self.metadata: Optional[WorkflowMetadata] = None
        self._initialize()

    @abstractmethod
    def _initialize(self):
        """初始化工作流（子类实现）"""
        pass

    @abstractmethod
    def get_input_schema(self) -> Type[BaseModel]:
        """获取输入Schema"""
        pass

    @abstractmethod
    def get_output_schema(self) -> Type[BaseModel]:
        """获取输出Schema"""
        pass

    @abstractmethod
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行工作流
        Returns: 执行结果
        """
        pass

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        运行工作流（带错误处理和日志）
        """
        import time
        import uuid

        # 初始化元数据
        self.metadata = WorkflowMetadata(
            workflow_id=str(uuid.uuid4()),
            workflow_name=self.config.name,
            start_time=time.time()
        )

        logger.info(f"[工作流] 开始执行: {self.config.name}")

        try:
            self.metadata.status = "running"

            # 执行工作流
            result = self.execute(input_data)

            self.metadata.status = "completed"
            self.metadata.end_time = time.time()

            logger.info(
                f"[工作流] 完成: {self.config.name}, "
                f"耗时: {self.metadata.duration:.2f}s"
            )

            # 添加执行元数据
            result['_metadata'] = self.metadata.to_dict()

            return result

        except Exception as e:
            self.metadata.status = "failed"
            self.metadata.end_time = time.time()
            self.metadata.errors.append(str(e))

            logger.error(f"[工作流] 执行失败: {self.config.name}, error: {e}")

            return {
                'success': False,
                'error': str(e),
                '_metadata': self.metadata.to_dict()
            }

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """验证输入"""
        try:
            schema = self.get_input_schema()
            schema(**input_data)
            return True
        except Exception as e:
            logger.error(f"[工作流] 输入验证失败: {e}")
            return False

    def get_available_nodes(self) -> List[str]:
        """获取可用节点列表"""
        return []


class WorkflowNode(ABC):
    """
    工作流节点基类
    标准化节点设计
    """

    def __init__(self, name: str):
        self.name = name
        self._input_schema: Optional[Type[BaseModel]] = None
        self._output_schema: Optional[Type[BaseModel]] = None

    @abstractmethod
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理数据"""
        pass

    def set_schemas(
        self,
        input_schema: Type[BaseModel],
        output_schema: Type[BaseModel]
    ):
        """设置输入输出Schema"""
        self._input_schema = input_schema
        self._output_schema = output_schema

    def validate_and_process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """验证并处理"""
        if self._input_schema:
            try:
                validated = self._input_schema(**input_data)
                input_data = validated.model_dump()
            except Exception as e:
                raise ValueError(f"节点 {self.name} 输入验证失败: {e}")

        result = self.process(input_data)

        if self._output_schema and result:
            try:
                validated = self._output_schema(**result)
                result = validated.model_dump()
            except Exception as e:
                raise ValueError(f"节点 {self.name} 输出验证失败: {e}")

        return result


# ==================== 工作流执行上下文 ====================

@dataclass
class WorkflowContext:
    """
    工作流执行上下文
    在节点间传递共享数据
    """
    workflow_id: str
    input_data: Dict[str, Any]
    state: Dict[str, Any] = field(default_factory=dict)
    cache: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def set_state(self, key: str, value: Any):
        """设置状态"""
        self.state[key] = value

    def get_state(self, key: str, default: Any = None) -> Any:
        """获取状态"""
        return self.state.get(key, default)

    def set_cache(self, key: str, value: Any, ttl: int = 3600):
        """设置缓存"""
        import time
        self.cache[key] = {
            'value': value,
            'expires_at': time.time() + ttl
        }

    def get_cache(self, key: str) -> Any:
        """获取缓存"""
        if key in self.cache:
            import time
            entry = self.cache[key]
            if time.time() < entry['expires_at']:
                return entry['value']
            else:
                del self.cache[key]
        return None
