"""智能VL引擎路由器

链式调用VL引擎：
1. MiniCPM-o 8B（最先进的VL理解，需vLLM/API）
2. Fallback VL（现有管线）
"""

import logging
from typing import Optional, Dict, Any, List, Tuple

from utils.vl_engines.base import BaseVLEngine, VLResult
from utils.vl_engines.minicpm_vl import MiniCPMVLEngine
from utils.vl_engines.fallback_vl import FallbackVLEngine

logger = logging.getLogger(__name__)


class SmartVLEngine(BaseVLEngine):
    """智能VL引擎路由器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or {}
        vl_cfg = self._config.get("vl_engines", {})

        self._engines: List[Tuple[str, BaseVLEngine, float]] = []

        # 1. MiniCPM-o（最高优先级）
        minicpm_cfg = vl_cfg.get("minicpm_o", {})
        minicpm = MiniCPMVLEngine(minicpm_cfg)
        self._engines.append(("minicpm_o", minicpm, 0.80))

        # 2. Fallback（始终可用）
        fb_cfg = vl_cfg.get("fallback", {})
        fallback = FallbackVLEngine(fb_cfg)
        self._engines.append(("fallback", fallback, 0.0))

        self._min_confidence = self._config.get("min_confidence", 0.3)

    @property
    def name(self) -> str:
        available = [e.name for _, e, _ in self._engines if e.is_available()]
        return f"SmartVL[{'|'.join(available)}]"

    def is_available(self) -> bool:
        return True

    def understand(self, image_url: str, prompt: str = "",
                   ocr_hint: str = "", options: Optional[Dict[str, Any]] = None) -> VLResult:
        opts = {**(options or {})}
        errors = []

        for name, engine, min_conf in self._engines:
            if not engine.is_available():
                logger.info(f"  ↪ {name}: unavailable, skip")
                continue
            try:
                logger.info(f"  → trying {name} ({engine.name})")
                result = engine.understand(image_url, prompt=prompt, ocr_hint=ocr_hint, options=opts)
                if result.success and result.confidence >= min_conf and len(result.structured_data) > 0:
                    logger.info(f"  ✓ {name}: success (conf={result.confidence:.2f}, fields={len(result.detected_fields)})")
                    result.metadata["engine_chain"] = name
                    result.metadata["fallback_chain"] = errors
                    return result
                else:
                    reason = f"low_conf({result.confidence:.2f})/empty" if not result.success else "empty_data"
                    logger.info(f"  ✗ {name}: {reason}")
                    errors.append({name: reason})
            except Exception as e:
                logger.warning(f"  ✗ {name}: exception {e}")
                errors.append({name: str(e)})

        error_msg = f"All VL engines failed: {errors}"
        logger.error(error_msg)
        return VLResult(
            structured_data={}, raw_response="", confidence=0.0,
            engine_name=self.name, success=False,
            metadata={"errors": errors}
        )

    def get_engine_status(self) -> Dict[str, Any]:
        return {
            name: {"available": e.is_available(), "name": e.name}
            for name, e, _ in self._engines
        }