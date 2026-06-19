"""通用自定义OCR模型适配器

支持任何兼容OpenAI API的模型端点（vLLM / Ollama / TGI / 自部署等）。
用户可在配置文件中定义任意多个自定义OCR模型，系统按优先级链式调用。

配置示例:
  "custom_engines": [
    {
      "name": "my-vllm-ocr",
      "endpoint": "https://my-server/v1/chat/completions",
      "model": "my-ocr-model",
      "api_key": "sk-xxx",
      "priority": 0,
      "prompt": "请识别图中的文字，返回纯文本"
    }
  ]

优先级规则：priority值越小优先级越高（0 > 1 > 2）
"""

import json, os, time, logging, base64, io
from typing import Optional, Dict, Any, List
from urllib.request import Request, urlopen
from urllib.error import URLError

from utils.ocr_engines.base import BaseOCREngine, OCRResult

logger = logging.getLogger(__name__)

DEFAULT_OCR_PROMPT = (
    "请仔细识别这张图片中的所有文字内容，包括中文、英文、数字和符号。"
    "保持原始排版和换行，不要添加任何额外解释。"
    "如果是产品包装，请尽量还原标签上的所有文字。"
)


class CustomOCREngine(BaseOCREngine):
    """通用自定义OCR模型适配器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or {}
        self._name = self._config.get("name", "custom-ocr")
        self._endpoint = self._config.get("endpoint", "")
        self._api_key = self._config.get("api_key", os.getenv("CUSTOM_MODEL_API_KEY", ""))
        self._model = self._config.get("model", "")
        self._prompt = self._config.get("prompt", DEFAULT_OCR_PROMPT)
        self._priority = self._config.get("priority", 99)
        self._max_tokens = self._config.get("max_tokens", 4096)
        self._temperature = self._config.get("temperature", 0.1)
        # 记录上一次请求失败的时间，用于指数退避
        self._last_fail_ts = 0.0
        self._retry_count = 0

    @property
    def name(self) -> str:
        return f"Custom[{self._name}]"

    @property
    def priority(self) -> int:
        return self._priority

    def is_available(self) -> bool:
        if not self._endpoint or not self._model:
            return False
        # 如果连续失败超过3次, 暂时标记为不可用(10分钟冷却)
        if self._retry_count >= 3:
            cooldown = time.time() - self._last_fail_ts
            if cooldown < 600:
                logger.info(f"  {self._name}: cooling down ({int(cooldown)}s/600s)")
                return False
            self._retry_count = 0
        return True

    def _encode_image(self, image_url: str) -> str:
        """图片编码：支持URL直传和base64"""
        if image_url.startswith("data:"):
            return image_url
        if image_url.startswith("http"):
            try:
                req = Request(image_url, headers={"User-Agent": "PackCV/1.0"})
                with urlopen(req, timeout=30) as resp:
                    data = resp.read()
                return "data:image/jpeg;base64," + base64.b64encode(data).decode()
            except Exception as e:
                logger.warning(f"  {self._name}: download image failed: {e}")
                raise
        return image_url

    def recognize(self, image_url: str, options: Optional[Dict[str, Any]] = None) -> OCRResult:
        start = time.time()
        opts = {**(options or {})}

        try:
            base64_image = self._encode_image(image_url)
            prompt = opts.get("prompt", self._prompt)

            payload = {
                "model": self._model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": base64_image}}
                        ]
                    }
                ],
                "max_tokens": opts.get("max_tokens", self._max_tokens),
                "temperature": opts.get("temperature", self._temperature)
            }

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}"
            }

            logger.info(f"  {self._name}: calling {self._endpoint} (model={self._model})")
            req = Request(
                self._endpoint,
                data=json.dumps(payload).encode(),
                headers=headers,
                method="POST"
            )
            with urlopen(req, timeout=120) as resp:
                response_data = json.loads(resp.read().decode())

            # 解析OpenAI格式响应
            raw_text = ""
            choices = response_data.get("choices", [])
            if choices:
                raw_text = choices[0].get("message", {}).get("content", "")
                if not raw_text:
                    raw_text = choices[0].get("text", "")

            elapsed = (time.time() - start) * 1000

            if raw_text.strip():
                self._retry_count = 0
                confidence = min(0.95, 0.6 + len(raw_text.strip()) * 0.001)
                logger.info(f"  ✓ {self._name}: {len(raw_text)} chars in {elapsed:.0f}ms")
                return OCRResult(
                    raw_text=raw_text.strip(),
                    confidence=confidence,
                    language="auto",
                    engine_name=self.name,
                    processing_time_ms=elapsed,
                    success=True,
                    metadata={"model": self._model, "endpoint": self._endpoint}
                )

            self._retry_count += 1
            self._last_fail_ts = time.time()
            return OCRResult(
                raw_text="", confidence=0.0, engine_name=self.name,
                success=False, metadata={"error": "empty response"}
            )

        except Exception as e:
            self._retry_count += 1
            self._last_fail_ts = time.time()
            elapsed = (time.time() - start) * 1000
            logger.warning(f"  ✗ {self._name}: {e}")
            return OCRResult(
                raw_text="", confidence=0.0, engine_name=self.name,
                success=False, metadata={"error": str(e)}
            )