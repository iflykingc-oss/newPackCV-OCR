"""降级OCR引擎 - 包装现有OCR工具"""

import time, logging
from typing import Optional, Dict, Any

from utils.ocr_engines.base import BaseOCREngine, OCRResult

logger = logging.getLogger(__name__)


class FallbackOCREngine(BaseOCREngine):
    """降级OCR引擎 - 使用目前管线中的OCR工具"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or {}
        self._engine_type = self._config.get("engine_type", "builtin")
        self._ocr_engine = None

    @property
    def name(self) -> str:
        return f"FallbackOCR({self._engine_type})"

    def is_available(self) -> bool:
        return True  # 降级引擎始终可用

    def _get_engine(self):
        """延迟加载OCR引擎"""
        if self._ocr_engine is None:
            from tools.ocr_tool import OCREngine
            self._ocr_engine = OCREngine(engine_type=self._engine_type)
        return self._ocr_engine

    def recognize(self, image_url: str, options: Optional[Dict[str, Any]] = None) -> OCRResult:
        start = time.time()
        try:
            engine = self._get_engine()
            result = engine.recognize(image_url)
            elapsed = (time.time() - start) * 1000
            raw_text = result.get("raw_text", "")
            confidence = result.get("confidence", 0.5)
            return OCRResult(
                raw_text=raw_text,
                confidence=confidence,
                language=result.get("language", "auto"),
                engine_name=self.name,
                processing_time_ms=elapsed,
                regions=result.get("regions", []),
                metadata={"engine_type": self._engine_type}
            )
        except Exception as e:
            logger.warning(f"FallbackOCR error: {e}")
            elapsed = (time.time() - start) * 1000
            return OCRResult(
                raw_text="", confidence=0.0, engine_name=self.name,
                processing_time_ms=elapsed,
                success=False, metadata={"error": str(e)}
            )