"""OCR引擎抽象基类"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class OCRResult(BaseModel):
    """OCR识别结果统一格式"""
    raw_text: str = Field(default="", description="识别到的原始文本")
    confidence: float = Field(default=0.0, description="全局置信度 0-1")
    language: str = Field(default="", description="检测到的语言")
    engine_name: str = Field(default="", description="使用的引擎名称")
    processing_time_ms: float = Field(default=0.0, description="处理耗时毫秒")
    regions: list = Field(default=[], description="文本区域列表")
    metadata: Dict[str, Any] = Field(default={}, description="引擎特有元数据")
    success: bool = Field(default=True, description="识别是否成功")


class BaseOCREngine(ABC):
    """OCR引擎抽象基类"""

    @abstractmethod
    def recognize(self, image_url: str, options: Optional[Dict[str, Any]] = None) -> OCRResult:
        """识别图片中的文字"""
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
        """返回引擎元数据"""
        return {"name": self.name, "available": self.is_available()}