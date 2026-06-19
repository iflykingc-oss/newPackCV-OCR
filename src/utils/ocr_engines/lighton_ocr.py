"""LightOnOCR-2-1B 适配器

支持两种模式：
1. vLLM/OpenAI兼容API → 通过标准vLLM endpoints
2. HuggingFace Inference API → 使用HF_TOKEN
"""

import json, os, time, logging, base64, io
from typing import Optional, Dict, Any
from urllib.request import Request, urlopen
from urllib.error import URLError

from utils.ocr_engines.base import BaseOCREngine, OCRResult

logger = logging.getLogger(__name__)


class LightOnOCREngine(BaseOCREngine):
    """LightOnOCR-2-1B 适配器"""

    MODE_VLLM = "vllm"
    MODE_HF_API = "hf_api"
    MODE_DISABLED = "disabled"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or {}
        self._mode = self._config.get("mode", self.MODE_DISABLED)
        self._endpoint = self._config.get("endpoint", "")
        self._hf_token = self._config.get("hf_token", os.getenv("HF_TOKEN", ""))
        self._model_id = self._config.get("model_id", "lightonai/LightOnOCR-2-1B")
        self._max_tokens = self._config.get("max_tokens", 4096)
        self._temperature = self._config.get("temperature", 0.2)

    @property
    def name(self) -> str:
        return f"LightOnOCR-2-1B({self._mode})"

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
                success=False, metadata={"error": f"LightOnOCR mode={self._mode} not available"}
            )

    # ── vLLM / OpenAI 兼容 API ──────────────────────────────────

    def _recognize_vllm(self, image_url: str, options: Dict[str, Any]) -> OCRResult:
        """调用 vLLM 服务的 chat/completions 接口"""
        import requests

        max_tokens = options.get("max_tokens", self._max_tokens)
        temperature = options.get("temperature", self._temperature)
        timeout = options.get("timeout", 60)

        # 构造消息：支持URL直接传递，不支持URL的话需要下载后转base64
        msg_content = [{"type": "image_url", "image_url": {"url": image_url}}]

        payload = {
            "model": self._model_id,
            "messages": [{"role": "user", "content": msg_content}],
            "max_tokens": max_tokens,
            "temperature": temperature,
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
                confidence=0.85, language="auto",
                engine_name=self.name, processing_time_ms=elapsed,
                metadata={"model": self._model_id, "tokens": data.get("usage", {})}
            )
        except Exception as e:
            logger.warning(f"LightOnOCR vLLM error: {e}")
            return OCRResult(
                raw_text="", confidence=0.0, engine_name=self.name,
                success=False, metadata={"error": str(e)}
            )

    # ── HuggingFace Inference API ────────────────────────────────

    def _recognize_hf_api(self, image_url: str, options: Dict[str, Any]) -> OCRResult:
        """通过 HF Inference API 调用"""
        max_tokens = options.get("max_tokens", self._max_tokens)
        timeout = options.get("timeout", 120)

        # 需要先把图片下载下来转base64
        image_data = self._download_image(image_url, timeout)
        if image_data is None:
            return OCRResult(
                raw_text="", confidence=0.0, engine_name=self.name,
                success=False, metadata={"error": "download image failed"}
            )
        b64 = base64.b64encode(image_data).decode("utf-8")
        data_uri = f"data:image/jpeg;base64,{b64}"

        url = f"https://api-inference.huggingface.co/models/{self._model_id}/v1/chat/completions"
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
            "temperature": 0.2,
        }

        try:
            import requests
            resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            elapsed = (time.time() - start) * 1000
            return OCRResult(
                raw_text=text.strip(),
                confidence=0.8, language="auto",
                engine_name=self.name, processing_time_ms=elapsed,
                metadata={"model": self._model_id, "mode": "hf_api"}
            )
        except Exception as e:
            logger.warning(f"LightOnOCR HF API error: {e}")
            return OCRResult(
                raw_text="", confidence=0.0, engine_name=self.name,
                success=False, metadata={"error": str(e)}
            )

    def _download_image(self, url: str, timeout: int = 30) -> Optional[bytes]:
        import requests
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp.content
        except Exception as e:
            logger.warning(f"Download image failed: {e}")
            return None