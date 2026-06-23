"""智能OCR引擎路由器

按优先级链式调用多个OCR引擎：
1. [自定义引擎] priority=0（最高，用户自部署模型）
2. Unlimited-OCR（priority=5，百度SOTA长文档解析）
3. LightOnOCR-2-1B（priority=10，1B参数）
4. DeepSeek-OCR（priority=20，3B参数）
5. Fallback（始终可用，保底）

策略：
- 按priority升序调用（0最高）
- 高优先级成功（confidence >= min_conf）→ 直接返回
- 失败/不可用 → 自动降级到下一个
"""

import logging
from typing import Optional, Dict, Any, List, Tuple

from utils.ocr_engines.base import BaseOCREngine, OCRResult
from utils.ocr_engines.lighton_ocr import LightOnOCREngine
from utils.ocr_engines.deepseek_ocr import DeepSeekOCREngine
from utils.ocr_engines.unlimited_ocr import UnlimitedOCREngine
from utils.ocr_engines.fallback_ocr import FallbackOCREngine
from utils.ocr_engines.custom_ocr import CustomOCREngine

logger = logging.getLogger(__name__)


class SmartOCREngine(BaseOCREngine):
    """智能OCR引擎路由器 - 支持自定义引擎的最高优先级"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or {}
        ocr_cfg = self._config.get("ocr_engines", {})

        # 构建引擎链: List[(engine_id, engine_instance, min_confidence, priority)]
        self._engines: List[Tuple[str, BaseOCREngine, float, int]] = []

        # 1. 自定义引擎（最高优先级，按priority升序）
        custom_list: list = ocr_cfg.get("custom_engines", []) or []
        for ce_cfg in custom_list:
            name = ce_cfg.get("name", "unnamed")
            priority = ce_cfg.get("priority", 99)
            kwargs = {"name": name, "priority": priority}
            engine = CustomOCREngine({**ce_cfg, **kwargs})
            min_conf = ce_cfg.get("min_confidence", 0.4)
            self._engines.append((f"custom_{name}", engine, min_conf, priority))
            logger.info(f"  SmartOCR: loaded custom engine [{name}] (priority={priority})")

        # 2. LightOnOCR-2-1B
        lighton_cfg = ocr_cfg.get("lighton_ocr", {})
        lighton = LightOnOCREngine(lighton_cfg)
        self._engines.append(("lighton_ocr", lighton, 0.85, 10))

        # 3. Unlimited-OCR (百度SOTA, priority=5 → 排在LightOn之前)
        unlimited_cfg = ocr_cfg.get("unlimited_ocr", {})
        unlimited = UnlimitedOCREngine(unlimited_cfg)
        self._engines.append(("unlimited_ocr", unlimited, 0.85, 5))

        # 4. DeepSeek-OCR
        ds_cfg = ocr_cfg.get("deepseek_ocr", {})
        deepseek = DeepSeekOCREngine(ds_cfg)
        self._engines.append(("deepseek_ocr", deepseek, 0.80, 20))

        # 5. Fallback（始终可用，最低优先级）
        fb_cfg = ocr_cfg.get("fallback", {"engine_type": self._config.get("engine_type", "builtin")})
        fallback = FallbackOCREngine(fb_cfg)
        self._engines.append(("fallback", fallback, 0.0, 999))

        # 按priority升序排序
        self._engines.sort(key=lambda x: x[3])

        self._min_confidence = self._config.get("min_confidence", 0.3)

    @property
    def name(self) -> str:
        available = [e.name for _, e, _, _ in self._engines if e.is_available()]
        return f"SmartOCR[{'|'.join(available)}]"

    def is_available(self) -> bool:
        return True

    def recognize(self, image_url: str, options: Optional[Dict[str, Any]] = None) -> OCRResult:
        opts = {**(options or {})}
        errors = []

        for name, engine, min_conf, priority in self._engines:
            if not engine.is_available():
                logger.info(f"  ↪ {name}: unavailable, skip")
                continue
            try:
                logger.info(f"  → trying {name} ({engine.name}) [pri={priority}]")
                result = engine.recognize(image_url, opts)
                if result.success and result.confidence >= min_conf and len(result.raw_text.strip()) > 0:
                    logger.info(f"  ✓ {name}: success (conf={result.confidence:.2f}, len={len(result.raw_text)})")
                    result.metadata["engine_chain"] = name
                    result.metadata["fallback_chain"] = errors
                    return result
                else:
                    reason = f"low_conf({result.confidence:.2f}<{min_conf})" if result.confidence < min_conf else "empty_text"
                    logger.info(f"  ✗ {name}: {reason}")
                    errors.append({name: reason})
            except Exception as e:
                logger.warning(f"  ✗ {name}: exception {e}")
                errors.append({name: str(e)})

        logger.error(f"All OCR engines failed: {errors}")
        return OCRResult(
            raw_text="", confidence=0.0, engine_name=self.name,
            success=False, metadata={"errors": errors}
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