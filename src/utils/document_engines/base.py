"""文档解析引擎抽象基类 (面向 PDF/DOCX/PPTX/XLSX)

与 BaseOCREngine 不同，文档引擎处理的是"文档"而非"图片"，
输出结构化 Markdown/JSON 而非纯文本。
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class DocumentParseResult(BaseModel):
    """文档解析结果统一格式"""
    markdown: str = Field(default="", description="Markdown格式全文")
    tables: List[Dict[str, Any]] = Field(default=[], description="表格列表(HTML+coordinates)")
    images: List[Dict[str, Any]] = Field(default=[], description="嵌入图片列表")
    metadata: Dict[str, Any] = Field(default={
        "pages": 0,
        "has_tables": False,
        "has_formulas": False,
        "has_stamps": False,
        "language": ""
    }, description="文档元数据")
    engine_name: str = Field(default="", description="使用的引擎名称")
    processing_time_ms: float = Field(default=0.0, description="处理耗时")
    success: bool = Field(default=True, description="解析是否成功")
    error: str = Field(default="", description="错误信息")


class BaseDocumentEngine(ABC):
    """文档解析引擎抽象基类"""

    @abstractmethod
    def parse(self, file_url: str, file_type: str,
              options: Optional[Dict[str, Any]] = None) -> DocumentParseResult:
        """解析文档"""
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

    @property
    @abstractmethod
    def supported_formats(self) -> List[str]:
        """支持的格式列表"""
        ...

    def can_handle(self, file_type: str) -> bool:
        """是否能处理该格式"""
        return file_type.lower() in self.supported_formats