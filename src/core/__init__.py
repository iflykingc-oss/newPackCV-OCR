# -*- coding: utf-8 -*-
"""
Core模块接口定义
三层架构核心接口规范：
- CV视觉层：图像预处理、目标检测、ROI裁切
- OCR识别层：多引擎统一调度
- 融合决策层：规则引擎、LLM条件触发
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


# ==================== 通用数据结构 ====================

class ImageFormat(Enum):
    """图片格式枚举"""
    JPEG = "jpeg"
    PNG = "png"
    BGR = "bgr"
    RGB = "rgb"
    GRAY = "gray"


@dataclass
class BoundingBox:
    """边界框定义"""
    x1: float
    y1: float
    x2: float
    y2: float
    score: float = 0.0

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


@dataclass
class OBBBox:
    """有向边界框（OBB）定义"""
    corners: List[List[float]]  # 4个角点 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    score: float = 0.0
    class_id: str = ""
    label: str = ""


# ==================== CV视觉层接口 ====================

@dataclass
class ROIObject:
    """ROI区域对象"""
    roi_id: str
    bbox: BoundingBox | OBBBox
    crop_image: Optional[bytes] = None  # 裁切图二进制
    crop_path: Optional[str] = None    # 裁切图路径
    confidence: float = 0.0
    class_id: str = ""
    label: str = ""
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class CVPreprocessor(ABC):
    """CV预处理抽象基类"""

    @abstractmethod
    def preprocess(self, image_path: str) -> Dict[str, Any]:
        """
        图像预处理
        Returns: {
            'processed_image': bytes,
            'image_format': ImageFormat,
            'metadata': {...}
        }
        """
        pass

    @abstractmethod
    def denoise(self, image: bytes) -> bytes:
        """去噪处理"""
        pass

    @abstractmethod
    def enhance_contrast(self, image: bytes) -> bytes:
        """对比度增强"""
        pass

    @abstractmethod
    def correct_perspective(self, image: bytes) -> bytes:
        """透视矫正"""
        pass


class CVDetector(ABC):
    """CV检测抽象基类"""

    @abstractmethod
    def detect(self, image_path: str) -> List[ROIObject]:
        """
        目标检测
        Returns: ROI对象列表
        """
        pass

    @abstractmethod
    def detect_obb(self, image_path: str) -> List[ROIObject]:
        """
        有向边界框检测
        Returns: ROI对象列表
        """
        pass


class ROICropper(ABC):
    """ROI裁切抽象基类"""

    @abstractmethod
    def crop(self, image_path: str, roi_objects: List[ROIObject]) -> List[ROIObject]:
        """
        ROI裁切
        Returns: 包含裁切图的ROI对象列表
        """
        pass

    @abstractmethod
    def apply_nms(self, roi_objects: List[ROIObject], iou_threshold: float = 0.5) -> List[ROIObject]:
        """
        非极大值抑制
        """
        pass


# ==================== OCR识别层接口 ====================

@dataclass
class OCRTextResult:
    """OCR文本识别结果"""
    text: str
    confidence: float
    bbox: Optional[BoundingBox] = None
    field_type: Optional[str] = None  # 字段类型：expiry/batch/brand/...


@dataclass
class OCRResult:
    """OCR识别结果"""
    raw_text: str
    full_text: str  # 合并后的完整文本
    regions: List[OCRTextResult]
    engine: str  # 使用的引擎
    confidence: float  # 整体置信度
    field_confidences: Dict[str, float] = None  # 字段级置信度

    def __post_init__(self):
        if self.field_confidences is None:
            self.field_confidences = {}

    def get_field_confidence(self, field_type: str) -> float:
        """获取特定字段的置信度"""
        return self.field_confidences.get(field_type, self.confidence)


class OCREngine(ABC):
    """OCR引擎抽象基类"""

    @property
    @abstractmethod
    def name(self) -> str:
        """引擎名称"""
        pass

    @property
    @abstractmethod
    def priority(self) -> int:
        """优先级（数字越小优先级越高）"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """检查引擎是否可用"""
        pass

    @abstractmethod
    def recognize(self, image_path: str) -> OCRResult:
        """执行OCR识别"""
        pass


class OCRScheduler:
    """
    OCR调度器
    统一管理多OCR引擎，实现：
    - 引擎优先级配置
    - 健康检测与自动降级
    - 多引擎结果融合
    - 字段级置信度评分
    """

    def __init__(self):
        self.engines: List[OCREngine] = []
        self.current_engine: Optional[OCREngine] = None

    def register_engine(self, engine: OCREngine):
        """注册OCR引擎"""
        self.engines.append(engine)
        self.engines.sort(key=lambda x: x.priority)

    def get_available_engine(self) -> Optional[OCREngine]:
        """获取可用的最高优先级引擎"""
        for engine in self.engines:
            if engine.is_available():
                return engine
        return None

    def recognize(self, image_path: str) -> OCRResult:
        """
        执行OCR识别，自动降级
        """
        engine = self.get_available_engine()
        if not engine:
            raise RuntimeError("No OCR engine available")

        try:
            return engine.recognize(image_path)
        except Exception as e:
            # 尝试下一个引擎
            for next_engine in self.engines[self.engines.index(engine) + 1:]:
                try:
                    return next_engine.recognize(image_path)
                except:
                    continue
            raise RuntimeError(f"All OCR engines failed: {e}")


# ==================== 融合决策层接口 ====================

@dataclass
class FieldValidationResult:
    """字段校验结果"""
    field_name: str
    value: Any
    is_valid: bool
    confidence: float
    error_message: Optional[str] = None


@dataclass
class LLMTriggerCondition:
    """LLM触发条件"""
    condition_type: str  # 'low_confidence' | 'conflict' | 'format_error' | 'custom'
    threshold: float = 0.85
    field_types: List[str] = None  # 适用的字段类型
    custom_rule: Optional[str] = None

    def __post_init__(self):
        if self.field_types is None:
            self.field_types = []


class RuleEngine(ABC):
    """规则引擎抽象基类"""

    @abstractmethod
    def validate_expiry(self, production_date: str, expiry_date: str) -> FieldValidationResult:
        """效期校验"""
        pass

    @abstractmethod
    def validate_batch(self, batch_number: str) -> FieldValidationResult:
        """批号校验"""
        pass

    @abstractmethod
    def check_conflict(self, ocr_result: OCRResult, llm_result: Dict) -> bool:
        """冲突检测"""
        pass


class LLMDecisionMaker:
    """
    LLM决策器
    条件触发LLM调用，避免全量调用
    """

    def __init__(self):
        self.trigger_conditions: List[LLMTriggerCondition] = []

    def add_trigger(self, condition: LLMTriggerCondition):
        """添加触发条件"""
        self.trigger_conditions.append(condition)

    def should_trigger_llm(self, ocr_result: OCRResult, field_type: str) -> bool:
        """
        判断是否需要触发LLM
        仅在以下场景触发：
        1. 低置信度：核心字段置信度 < 阈值
        2. 冲突：多引擎结果不一致
        3. 格式错误：日期/数字格式不符
        """
        for condition in self.trigger_conditions:
            if condition.condition_type == 'low_confidence':
                if field_type in condition.field_types or not condition.field_types:
                    field_conf = ocr_result.get_field_confidence(field_type)
                    if field_conf < condition.threshold:
                        return True
        return False


@dataclass
class ProcessResult:
    """处理结果"""
    success: bool
    data: Any
    errors: List[str] = None
    warnings: List[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
        if self.metadata is None:
            self.metadata = {}
