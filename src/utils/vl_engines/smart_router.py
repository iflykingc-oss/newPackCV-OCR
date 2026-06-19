"""智能VL引擎路由器

按优先级链式调用VL引擎：
1. [自定义引擎] priority=0（最高，用户自部署模型）
2. MiniCPM-o（priority=10，8B参数）
3. Fallback（始终可用，保底）
"""

import logging
from typing import Optional, Dict, Any, List, Tuple

from utils.vl_engines.base import BaseVLEngine, VLResult
from utils.vl_engines.minicpm_vl import MiniCPMVLEngine
from utils.vl_engines.fallback_vl import FallbackVLEngine
from utils.vl_engines.custom_vl import CustomVLEngine

logger = logging.getLogger(__name__)


class SmartVLEngine(BaseVLEngine):
    """智能VL引擎路由器 - 支持自定义引擎的最高优先级"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or {}
        vl_cfg = self._config.get("vl_engines", {})

        self._engines: List[Tuple[str, BaseVLEngine, float, int]] = []

        # 1. 自定义VL引擎（最高优先级）
        custom_list: list = vl_cfg.get("custom_engines", []) or []
        for ce_cfg in custom_list:
            name = ce_cfg.get("name", "unnamed")
            priority = ce_cfg.get("priority", 99)
            engine = CustomVLEngine({**ce_cfg, "name": name, "priority": priority})
            min_conf = ce_cfg.get("min_confidence", 0.5)
            self._engines.append((f"custom_{name}", engine, min_conf, priority))
            logger.info(f"  SmartVL: loaded custom engine [{name}] (priority={priority})")

        # 2. MiniCPM-o
        minicpm_cfg = vl_cfg.get("minicpm_o", {})
        minicpm = MiniCPMVLEngine(minicpm_cfg)
        self._engines.append(("minicpm_o", minicpm, 0.80, 10))

        # 3. Fallback
        fb_cfg = vl_cfg.get("fallback", {})
        fallback = FallbackVLEngine(fb_cfg)
        self._engines.append(("fallback", fallback, 0.0, 999))

        # 按priority升序排序
        self._engines.sort(key=lambda x: x[3])

        self._min_confidence = self._config.get("min_confidence", 0.3)

    @property
    def name(self) -> str:
        available = [e.name for _, e, _, _ in self._engines if e.is_available()]
        return f"SmartVL[{'|'.join(available)}]"

    def is_available(self) -> bool:
        return True

    def understand(self, image_url: str, prompt: str = "",
                   ocr_hint: str = "", options: Optional[Dict[str, Any]] = None) -> VLResult:
        opts = {**(options or {})}
        errors = []

        for name, engine, min_conf, priority in self._engines:
            if not engine.is_available():
                logger.info(f"  ↪ {name}: unavailable, skip")
                continue
            try:
                logger.info(f"  → trying {name} ({engine.name}) [pri={priority}]")
                result = engine.understand(image_url, prompt=prompt, ocr_hint=ocr_hint, options=opts)
                if result.success and result.confidence >= min_conf and len(result.structured_data) > 0:
                    logger.info(f"  ✓ {name}: success (conf={result.confidence:.2f}, fields={len(result.detected_fields)})")
                    result.metadata["engine_chain"] = name
                    result.metadata["fallback_chain"] = errors
                    return result
                else:
                    reason = f"low_conf({result.confidence:.2f})/empty"
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
            name: {
                "available": e.is_available(),
                "name": e.name,
                "priority": p
            }
            for name, e, _, p in self._engines
        }