"""DeepSeek-OCR 适配器

支持两种模式：
1. vLLM/OpenAI兼容API → 通过标准vLLM endpoints
2. HuggingFace Inference API → 使用HF_TOKEN
"""

import os, time, logging, base64
from typing import Optional, Dict, Any

from utils.ocr_engines.base import BaseOCREngine, OCRResult

logger = logging.getLogger(__name__)


class DeepSeekOCREngine(BaseOCREngine):
    """DeepSeek-OCR 适配器（v1或v2）"""

    MODE_VLLM = "vllm"
    MODE_HF_API = "hf_api"
    MODE_DISABLED = "disabled"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or {}
        self._mode = self._config.get("mode", self.MODE_DISABLED)
        self._endpoint = self._config.get("endpoint", "")
        self._hf_token = self._config.get("hf_token", os.getenv("HF_TOKEN", ""))
        # 默认使用 DeepSeek-OCR-2（v2版本）
        self._model_id = self._config.get("model_id", "deepseek-ai/DeepSeek-OCR-2")
        self._max_tokens = self._config.get("max_tokens", 2048)
        self._num_visual_tokens = self._config.get("num_visual_tokens", 256)

    @property
    def name(self) -> str:
        return f"DeepSeek-OCR({self._mode})"

    def is_available(self) -> bool:
        if self._mode == self.MODE_VLLM:
            return bool(self._endpoint)
        elif self._mode == self.MODE_HF_API:
            return bool(self._hf_token)
        return False

    def recognize(self, image_url: str, options: Optional[Dict[str, Any]] = None) -> OCRResult:
        start = time.time()
        opts = {**(options or {})}

        if self._mode == self.MODE_VLLM:
            return self._recognize_vllm(image_url, opts)
        elif self._mode == self.MODE_HF_API:
            return self._recognize_hf_api(image_url, opts)
        else:
            return OCRResult(
                raw_text="", confidence=0.0, engine_name=self.name,
                success=False, metadata={"error": f"DeepSeek-OCR mode={self._mode} not available"}
            )

    # ── vLLM / OpenAI 兼容 API ──────────────────────────────────

    def _recognize_vllm(self, image_url: str, options: Dict[str, Any]) -> OCRResult:
        import requests
        max_tokens = options.get("max_tokens", self._max_tokens)
        timeout = options.get("timeout", 60)

        # DeepSeek-OCR uses the same OpenAI-compatible chat format
        msg_content = [{"type": "image_url", "image_url": {"url": image_url}}]

        payload = {
            "model": self._model_id,
            "messages": [{"role": "user", "content": msg_content}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }

        headers = {"Content-Type": "application/json"}
        if self._config.get("api_key"):
            headers["Authorization"] = f"Bearer {self._config['api_key']}"

        try:
            resp = requests.post(
                f"{self._endpoint.rstrip('/')}/v1/chat/completions",
                json=payload, headers=headers, timeout=timeout
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            elapsed = (time.time() - start) * 1000
            return OCRResult(
                raw_text=text.strip(),
                confidence=0.9, language="auto",
                engine_name=self.name, processing_time_ms=elapsed,
                metadata={"model": self._model_id, "tokens": data.get("usage", {})}
            )
        except Exception as e:
            logger.warning(f"DeepSeek-OCR vLLM error: {e}")
            return OCRResult(
                raw_text="", confidence=0.0, engine_name=self.name,
                success=False, metadata={"error": str(e)}
            )

    # ── HuggingFace Inference API ────────────────────────────────

    def _recognize_hf_api(self, image_url: str, options: Dict[str, Any]) -> OCRResult:
        import requests
        max_tokens = options.get("max_tokens", self._max_tokens)
        timeout = options.get("timeout", 120)

        # 下载图片转base64
        try:
            resp_img = requests.get(image_url, timeout=30)
            resp_img.raise_for_status()
            b64 = base64.b64encode(resp_img.content).decode("utf-8")
            data_uri = f"data:image/jpeg;base64,{b64}"
        except Exception as e:
            return OCRResult(
                raw_text="", confidence=0.0, engine_name=self.name,
                success=False, metadata={"error": f"download image: {e}"}
            )

        hf_url = f"https://api-inference.huggingface.co/models/{self._model_id}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._hf_token}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self._model_id,
            "messages": [{
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": data_uri}}]
            }],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }

        try:
            resp = requests.post(hf_url, json=payload, headers=headers, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            elapsed = (time.time() - start) * 1000
            return OCRResult(
                raw_text=text.strip(),
                confidence=0.85, language="auto",
                engine_name=self.name, processing_time_ms=elapsed,
                metadata={"model": self._model_id, "mode": "hf_api"}
            )
        except Exception as e:
            logger.warning(f"DeepSeek-OCR HF API error: {e}")
            return OCRResult(
                raw_text="", confidence=0.0, engine_name=self.name,
                success=False, metadata={"error": str(e)}
            )