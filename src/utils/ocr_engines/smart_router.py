"""智能OCR引擎路由器

按优先级链式调用多个OCR引擎：
1. LightOnOCR-2-1B（最快最准，1B参数）
2. DeepSeek-OCR（强在文档级理解，3B参数）
3. Fallback OCR（现有Tesseract/PaddleOCR管线）

策略：
- 高优先级引擎成功 → 直接返回
- 高优先级引擎失败/不可用 → 自动降级
- 降级条件：confidence < min_confidence 或 success=False
"""

import logging
from typing import Optional, Dict, Any, List, Tuple

from utils.ocr_engines.base import BaseOCREngine, OCRResult
from utils.ocr_engines.lighton_ocr import LightOnOCREngine
from utils.ocr_engines.deepseek_ocr import DeepSeekOCREngine
from utils.ocr_engines.fallback_ocr import FallbackOCREngine

logger = logging.getLogger(__name__)


class SmartOCREngine(BaseOCREngine):
    """智能OCR引擎路由器 - 链式调用+自动降级"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or {}
        ocr_cfg = self._config.get("ocr_engines", {})

        # 构建引擎链
        self._engines: List[Tuple[str, BaseOCREngine, float]] = []

        # 1. LightOnOCR-2-1B（最高优先级）
        lighton_cfg = ocr_cfg.get("lighton_ocr", {})
        lighton = LightOnOCREngine(lighton_cfg)
        self._engines.append(("lighton_ocr", lighton, 0.85))

        # 2. DeepSeek-OCR（次高优先级）
        ds_cfg = ocr_cfg.get("deepseek_ocr", {})
        deepseek = DeepSeekOCREngine(ds_cfg)
        self._engines.append(("deepseek_ocr", deepseek, 0.80))

        # 3. Fallback（始终可用）
        fb_cfg = ocr_cfg.get("fallback", {"engine_type": self._config.get("engine_type", "builtin")})
        fallback = FallbackOCREngine(fb_cfg)
        self._engines.append(("fallback", fallback, 0.0))

        self._min_confidence = self._config.get("min_confidence", 0.3)

    @property
    def name(self) -> str:
        available = [e.name for _, e, _ in self._engines if e.is_available()]
        return f"SmartOCR[{'|'.join(available)}]"

    def is_available(self) -> bool:
        return True  # fallback始终可用

    def recognize(self, image_url: str, options: Optional[Dict[str, Any]] = None) -> OCRResult:
        opts = {**(options or {})}
        errors = []

        for name, engine, min_conf in self._engines:
            if not engine.is_available():
                logger.info(f"  ↪ {name}: unavailable, skip")
                continue

            try:
                logger.info(f"  → trying {name} ({engine.name})")
                result = engine.recognize(image_url, opts)
                if result.success and result.confidence >= min_conf and len(result.raw_text.strip()) > 0:
                    logger.info(f"  ✓ {name}: success (conf={result.confidence:.2f}, len={len(result.raw_text)})")
                    result.metadata["engine_chain"] = name
                    result.metadata["fallback_chain"] = errors
                    return result
                else:
                    reason = f"low_confidence({result.confidence:.2f}<{min_conf})" if not result.success else f"empty_text"
                    logger.info(f"  ✗ {name}: {reason}")
                    errors.append({name: reason})
            except Exception as e:
                logger.warning(f"  ✗ {name}: exception {e}")
                errors.append({name: str(e)})

        # 所有引擎都失败
        logger.error(f"All OCR engines failed: {errors}")
        return OCRResult(
            raw_text="", confidence=0.0, engine_name=self.name,
            success=False, metadata={"errors": errors}
        )

    def get_engine_status(self) -> Dict[str, Any]:
        """返回各引擎状态"""
        return {
            name: {"available": e.is_available(), "name": e.name}
            for name, e, _ in self._engines
        }