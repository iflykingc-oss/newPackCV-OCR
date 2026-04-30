# -*- coding: utf-8 -*-
"""
OCR识别层模块
提供多引擎统一调度能力
"""

from core.ocr.ocr_scheduler import (
    OCREngine,
    OCRResult,
    OCRScheduler,
    OCRTextResult,
    TesseractEngine,
    EasyOCREngine,
    PaddleOCREngine,
    create_ocr_scheduler,
    EngineHealth
)

__all__ = [
    'OCREngine',
    'OCRResult',
    'OCRScheduler',
    'OCRTextResult',
    'TesseractEngine',
    'EasyOCREngine',
    'PaddleOCREngine',
    'create_ocr_scheduler',
    'EngineHealth'
]
