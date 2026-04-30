# -*- coding: utf-8 -*-
"""
OCR调度器实现
多引擎统一调度、优先级配置、健康检测、自动降级
"""

import os
import time
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from core import (
    OCREngine, OCRResult, OCRScheduler,
    OCRTextResult, BoundingBox
)


logger = logging.getLogger(__name__)


@dataclass
class EngineHealth:
    """引擎健康状态"""
    name: str
    is_healthy: bool = True
    last_check_time: float = 0
    consecutive_failures: int = 0
    avg_response_time: float = 0
    total_requests: int = 0


class TesseractEngine(OCREngine):
    """Tesseract OCR引擎实现"""

    def __init__(self, timeout: int = 30):
        self._timeout = timeout
        self._health = EngineHealth(name="tesseract")
        self._check_availability()

    @property
    def name(self) -> str:
        return "tesseract"

    @property
    def priority(self) -> int:
        return 1  # 最高优先级，无需下载模型

    def is_available(self) -> bool:
        return self._health.is_healthy

    def _check_availability(self):
        """检查引擎可用性"""
        try:
            import pytesseract
            import shutil
            # 检查tesseract命令是否存在
            path = shutil.which('tesseract')
            if path:
                self._health.is_healthy = True
                logger.info("[OCR调度] Tesseract引擎可用")
            else:
                self._health.is_healthy = False
                logger.warning("[OCR调度] Tesseract命令未找到")
        except ImportError:
            self._health.is_healthy = False
            logger.warning("[OCR调度] pytesseract未安装")

    def recognize(self, image_path: str) -> OCRResult:
        """执行Tesseract识别"""
        start_time = time.time()

        try:
            import pytesseract
            from PIL import Image

            # 打开图片
            img = Image.open(image_path)

            # 执行识别
            text = pytesseract.image_to_string(img, lang='chi_sim+eng')

            # 获取详细数据
            data = pytesseract.image_to_data(img, lang='chi_sim+eng', output_type=pytesseract.Output.DICT)

            # 解析区域
            regions = []
            confidences = []
            for i in range(len(data['text'])):
                t = data['text'][i]
                conf = data['conf'][i]

                if t.strip() and int(conf) > 0:
                    regions.append(OCRTextResult(
                        text=t,
                        confidence=int(conf) / 100.0,
                        bbox=BoundingBox(
                            x1=data['left'][i],
                            y1=data['top'][i],
                            x2=data['left'][i] + data['width'][i],
                            y2=data['top'][i] + data['height'][i],
                            score=int(conf) / 100.0
                        )
                    ))
                    confidences.append(int(conf) / 100.0)

            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

            # 更新健康状态
            self._update_health(time.time() - start_time, success=True)

            return OCRResult(
                raw_text=text.strip(),
                full_text=text.strip(),
                regions=regions,
                engine="tesseract",
                confidence=avg_confidence,
                field_confidences={'default': avg_confidence}
            )

        except Exception as e:
            self._update_health(0, success=False)
            raise RuntimeError(f"Tesseract识别失败: {e}")

    def _update_health(self, response_time: float, success: bool):
        """更新健康状态"""
        self._health.last_check_time = time.time()
        self._health.total_requests += 1

        if success:
            self._health.consecutive_failures = 0
            self._health.is_healthy = True
            # 更新平均响应时间
            if self._health.avg_response_time == 0:
                self._health.avg_response_time = response_time
            else:
                self._health.avg_response_time = (
                    self._health.avg_response_time * 0.7 + response_time * 0.3
                )
        else:
            self._health.consecutive_failures += 1
            # 连续失败3次标记为不健康
            if self._health.consecutive_failures >= 3:
                self._health.is_healthy = False


class EasyOCREngine(OCREngine):
    """EasyOCR引擎实现"""

    def __init__(self, timeout: int = 120):
        self._timeout = timeout
        self._reader = None
        self._health = EngineHealth(name="easyocr")

    @property
    def name(self) -> str:
        return "easyocr"

    @property
    def priority(self) -> int:
        return 2  # 需要下载模型，优先级次之

    def is_available(self) -> bool:
        return self._health.is_healthy and self._reader is not None

    def _ensure_reader(self):
        """确保reader已初始化"""
        if self._reader is None:
            import easyocr
            self._reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)
            self._health.is_healthy = True
            logger.info("[OCR调度] EasyOCR引擎初始化完成")

    def recognize(self, image_path: str) -> OCRResult:
        """执行EasyOCR识别"""
        start_time = time.time()

        try:
            self._ensure_reader()
            result = self._reader.readtext(image_path)

            # 解析结果
            regions = []
            confidences = []
            full_text_parts = []

            for item in result:
                bbox = item[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                text = item[1]
                conf = float(item[2])

                if text and text.strip():
                    regions.append(OCRTextResult(
                        text=text,
                        confidence=conf,
                        bbox=BoundingBox(
                            x1=min(p[0] for p in bbox),
                            y1=min(p[1] for p in bbox),
                            x2=max(p[0] for p in bbox),
                            y2=max(p[1] for p in bbox),
                            score=conf
                        )
                    ))
                    confidences.append(conf)
                    full_text_parts.append(text)

            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            full_text = '\n'.join(full_text_parts)

            # 更新健康状态
            self._update_health(time.time() - start_time, success=True)

            return OCRResult(
                raw_text=full_text.strip(),
                full_text=full_text.strip(),
                regions=regions,
                engine="easyocr",
                confidence=avg_confidence,
                field_confidences={'default': avg_confidence}
            )

        except Exception as e:
            self._update_health(0, success=False)
            raise RuntimeError(f"EasyOCR识别失败: {e}")

    def _update_health(self, response_time: float, success: bool):
        """更新健康状态"""
        self._health.last_check_time = time.time()
        self._health.total_requests += 1

        if success:
            self._health.consecutive_failures = 0
            self._health.is_healthy = True
            if self._health.avg_response_time == 0:
                self._health.avg_response_time = response_time
            else:
                self._health.avg_response_time = (
                    self._health.avg_response_time * 0.7 + response_time * 0.3
                )
        else:
            self._health.consecutive_failures += 1
            if self._health.consecutive_failures >= 3:
                self._health.is_healthy = False


# 注意：PaddleOCR已禁用，因其在首次使用时会自动下载大量模型（数百MB），导致长时间卡顿
# 当前仅支持Tesseract和EasyOCR两个引擎


def create_ocr_scheduler() -> OCRScheduler:
    """
    创建OCR调度器
    按优先级注册引擎：Tesseract > EasyOCR > PaddleOCR
    """
    scheduler = OCRScheduler()

    # 注册Tesseract（最高优先级，无需下载模型）
    try:
        tesseract = TesseractEngine()
        if tesseract.is_available():
            scheduler.register_engine(tesseract)
            logger.info("[OCR调度] Tesseract引擎已注册")
    except Exception as e:
        logger.warning(f"[OCR调度] Tesseract引擎注册失败: {e}")

    # 注册EasyOCR
    try:
        easyocr = EasyOCREngine()
        if easyocr.is_available():
            scheduler.register_engine(easyocr)
            logger.info("[OCR调度] EasyOCR引擎已注册")
        else:
            logger.info("[OCR调度] EasyOCR引擎待初始化（首次使用会下载模型）")
            scheduler.register_engine(easyocr)
    except Exception as e:
        logger.warning(f"[OCR调度] EasyOCR引擎注册失败: {e}")

    # 注意：PaddleOCR已禁用，因其在首次使用时会自动下载大量模型（数百MB），导致长时间卡顿
    # 如果需要启用PaddleOCR，请手动取消下面的注释
    # try:
    #     paddleocr = PaddleOCREngine()
    #     if paddleocr.is_available():
    #         scheduler.register_engine(paddleocr)
    #         logger.info("[OCR调度] PaddleOCR引擎已注册")
    # except Exception as e:
    #     logger.warning(f"[OCR调度] PaddleOCR引擎注册失败: {e}")

    return scheduler
