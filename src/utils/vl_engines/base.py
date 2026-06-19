"""VL引擎抽象基类"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class VLResult(BaseModel):
    """VL引擎理解结果"""
    structured_data: Dict[str, Any] = Field(default={}, description="结构化提取结果")
    raw_response: str = Field(default="", description="模型原始响应文本")
    confidence: float = Field(default=0.0, description="全局置信度 0-1")
    engine_name: str = Field(default="", description="使用的引擎名称")
    processing_time_ms: float = Field(default=0.0, description="处理耗时毫秒")
    detected_fields: List[str] = Field(default=[], description="成功提取的字段列表")
    metadata: Dict[str, Any] = Field(default={}, description="引擎特有元数据")
    success: bool = Field(default=True, description="识别是否成功")


class BaseVLEngine(ABC):
    """VL引擎抽象基类"""

    @abstractmethod
    def understand(self, image_url: str, prompt: str = "",
                   ocr_hint: str = "", options: Optional[Dict[str, Any]] = None) -> VLResult:
        """理解图片中的视觉内容"""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """检查引擎是否可用"""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """引擎名称"""
        ...

    def get_metadata(self) -> Dict[str, Any]:
        return {"name": self.name, "available": self.is_available()}