"""PackCV SDK - 数据模型"""

from enum import Enum
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime


class Scenario(str, Enum):
    """支持的场景类型"""
    PACKAGING = "packaging"
    FINANCE_RECEIPT = "finance_receipt"
    FINANCE_STATEMENT = "finance_statement"
    PHARMACEUTICAL = "pharmaceutical"
    CONTRACT = "contract"
    ID_CARD = "id_card"
    LOGISTICS = "logistics"
    GENERAL_DOCUMENT = "general_document"
    AUTO = "auto"  # 自动识别


class EngineTier(int, Enum):
    """引擎优先级 (数字越小优先级越高)"""
    CUSTOM = 0       # 用户自定义模型
    PADDLEOCR_VL = 1 # PaddleOCR-VL-1.6
    LIGHTON = 2      # LightOnOCR-2-1B
    DEEPSEEK = 3     # DeepSeek-OCR
    FALLBACK = 99    # Tesseract/PaddleOCR/RapidOCR


@dataclass
class ExtractRequest:
    """提取请求参数"""
    image: Optional[Union[str, bytes]] = None
    document: Optional[Union[str, bytes]] = None
    scenario: Scenario = Scenario.AUTO
    locale: str = "zh-CN"
    engine_tier: Optional[EngineTier] = None  # None=自动选择
    webhook_url: Optional[str] = None
    timeout: int = 30
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image": self.image,
            "document": self.document,
            "scenario": self.scenario.value if isinstance(self.scenario, Scenario) else self.scenario,
            "locale": self.locale,
            "engine_tier": self.engine_tier.value if self.engine_tier is not None else None,
            "webhook_url": self.webhook_url,
            "timeout": self.timeout,
            **self.extra,
        }


@dataclass
class ExtractResult:
    """提取结果"""
    request_id: str
    scenario: str
    fields: Dict[str, Any]
    confidence: float
    engine_used: str
    latency_ms: float
    raw_text: Optional[str] = None
    raw_markdown: Optional[str] = None
    bbox_data: Optional[List[Dict[str, Any]]] = None
    tables: Optional[List[Dict[str, Any]]] = None
    barcode_results: Optional[List[Dict[str, Any]]] = None
    stamp_results: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[str] = None

    def get(self, key: str, default: Any = None) -> Any:
        """获取字段值 (类似 dict.get)"""
        return self.fields.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_response(cls, data: Dict[str, Any]) -> "ExtractResult":
        return cls(
            request_id=data.get("request_id", ""),
            scenario=data.get("scenario", "general_document"),
            fields=data.get("fields", {}),
            confidence=data.get("confidence", 0.0),
            engine_used=data.get("engine_used", "unknown"),
            latency_ms=data.get("latency_ms", 0.0),
            raw_text=data.get("raw_text"),
            raw_markdown=data.get("raw_markdown"),
            bbox_data=data.get("bbox_data"),
            tables=data.get("tables"),
            barcode_results=data.get("barcode_results"),
            stamp_results=data.get("stamp_results"),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", datetime.utcnow().isoformat()),
        )


@dataclass
class BatchResult:
    """批量结果"""
    total: int
    succeeded: int
    failed: int
    results: List[Union[ExtractResult, Exception]]

    @property
    def success_rate(self) -> float:
        return self.succeeded / self.total if self.total > 0 else 0.0
