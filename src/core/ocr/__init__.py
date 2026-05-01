# -*- coding: utf-8 -*-
"""
OCR识别层模块
提供多引擎统一调度能力
注意：PaddleOCR已禁用（需要下载大量模型，导致长时间卡顿）
当前使用：Tesseract（主引擎）+ EasyOCR（备用）+ 内嵌OCR（兜底）
"""

from core.ocr.ocr_scheduler import (
    OCREngine,
    OCRResult,
    OCRScheduler,
    OCRTextResult,
    TesseractEngine,
    EasyOCREngine,
    create_ocr_scheduler,
    EngineHealth
)

# 内嵌OCR导入 - 使用绝对导入
try:
    from core.ocr.builtin_ocr import (
        BuiltinOCR,
        TextRegion,
        OCRResult as BuiltinOCRResult,
        recognize_text,
        extract_structured_info
    )
    _HAS_BUILTIN_OCR = True
except ImportError:
    _HAS_BUILTIN_OCR = False
    BuiltinOCR = None
    TextRegion = None
    BuiltinOCRResult = None
    recognize_text = None
    extract_structured_info = None

__all__ = [
    'OCREngine',
    'OCRResult',
    'OCRScheduler',
    'OCRTextResult',
    'TesseractEngine',
    'EasyOCREngine',
    'create_ocr_scheduler',
    'EngineHealth',
    'BuiltinOCR',
    'TextRegion',
    'BuiltinOCRResult',
    'recognize_text',
    'extract_structured_info'
]
