"""百度 Unlimited-OCR 适配器

基于 DeepSeek-OCR 架构的 SOTA 长文档解析模型，百度出品。
核心优势：一次性端到端解析，无需传统"检测→识别→拼接"流水线。

支持两种调用模式：
1. vLLM/SGLang/OpenAI兼容API → 推荐生产部署
2. HuggingFace Inference API → 快速验证

配置示例:
  "unlimited_ocr": {
    "mode": "vllm",
    "endpoint": "http://your-server:10000",
    "model_id": "baidu/Unlimited-OCR",
    "image_mode": "gundam",
    "api_key": "sk-xxx"
  }

image_mode 说明：
  - gundam: 切片模式(base_size=1024, image_size=640)，适合包装标签等单张图
  - base: 整图模式(base_size=1024, image_size=1024)，适合整页文档
"""

import os, time, logging, base64
from typing import Optional, Dict, Any

from utils.ocr_engines.base import BaseOCREngine, OCRResult

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = (
    "<image>document parsing. "
    "Please extract all text content from this image accurately, "
    "preserving the original layout and formatting. "
    "Output the recognized text directly without any explanation."
)


class UnlimitedOCREngine(BaseOCREngine):
    """百度 Unlimited-OCR 适配器 — SOTA 长文档一次性解析"""

    MODE_VLLM = "vllm"
    MODE_HF_API = "hf_api"
    MODE_DISABLED = "disabled"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or {}
        self._mode = self._config.get("mode", self.MODE_DISABLED)
        self._endpoint = self._config.get("endpoint", "")
        self._api_key = self._config.get("api_key", "")
        self._hf_token = self._config.get("hf_token", os.getenv("HF_TOKEN", ""))
        self._model_id = self._config.get("model_id", "baidu/Unlimited-OCR")
        self._max_tokens = self._config.get("max_tokens", 32768)
        self._image_mode = self._config.get("image_mode", "gundam")
        self._prompt = self._config.get("prompt", DEFAULT_PROMPT)
        self._ngram_size = self._config.get("ngram_size", 35)
        self._ngram_window = self._config.get("ngram_window", 128)

    @property
    def name(self) -> str:
        return f"Unlimited-OCR({self._mode})"

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
                success=False, metadata={"error": f"Unlimited-OCR mode={self._mode} not available"}
            )

    def _encode_image(self, image_url: str) -> str:
        if image_url.startswith("data:"):
            return image_url
        if image_url.startswith("http"):
            import requests
            try:
                resp = requests.get(image_url, timeout=30)
                resp.raise_for_status()
                ext = image_url.rsplit(".", 1)[-1].lower() if "." in image_url else "jpeg"
                mime = f"image/{'jpeg' if ext in ('jpg', 'jpeg') else ext}"
                b64 = base64.b64encode(resp.content).decode("utf-8")
                return f"data:{mime};base64,{b64}"
            except Exception as e:
                logger.warning(f"  Unlimited-OCR: download image failed: {e}")
                raise
        return image_url

    # ── vLLM / SGLang / OpenAI 兼容 API ──────────────────────────

    def _recognize_vllm(self, image_url: str, options: Dict[str, Any]) -> OCRResult:
        import requests
        max_tokens = options.get("max_tokens", self._max_tokens)
        timeout = options.get("timeout", 120)
        image_mode = options.get("image_mode", self._image_mode)
        prompt = options.get("prompt", self._prompt)

        base64_image = self._encode_image(image_url)

        payload = {
            "model": self._model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": base64_image}}
                    ]
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "skip_special_tokens": False,
        }

        # SGLang 特有参数：image_mode + custom logit processor
        if image_mode:
            payload["images_config"] = {"image_mode": image_mode}
        if self._ngram_size and self._ngram_window:
            payload["custom_logit_processor"] = "UnlimitedOCRNoRepeatNGramLogitProcessor"
            payload["custom_params"] = {
                "ngram_size": self._ngram_size,
                "window_size": self._ngram_window,
            }

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        try:
            resp = requests.post(
                f"{self._endpoint.rstrip('/')}/v1/chat/completions",
                json=payload, headers=headers, timeout=timeout
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            elapsed = (time.time() - start) * 1000

            confidence = 0.92 if len(text.strip()) > 50 else 0.85
            return OCRResult(
                raw_text=text.strip(),
                confidence=confidence, language="auto",
                engine_name=self.name, processing_time_ms=elapsed,
                metadata={
                    "model": self._model_id,
                    "image_mode": image_mode,
                    "tokens": data.get("usage", {}),
                    "source": "baidu_unlimited_ocr_sota"
                }
            )
        except Exception as e:
            logger.warning(f"  Unlimited-OCR vLLM error: {e}")
            return OCRResult(
                raw_text="", confidence=0.0, engine_name=self.name,
                success=False, metadata={"error": str(e)}
            )

    # ── HuggingFace Inference API ────────────────────────────────

    def _recognize_hf_api(self, image_url: str, options: Dict[str, Any]) -> OCRResult:
        import requests
        max_tokens = options.get("max_tokens", self._max_tokens)
        timeout = options.get("timeout", 120)
        prompt = options.get("prompt", self._prompt)

        try:
            base64_image = self._encode_image(image_url)
        except Exception as e:
            return OCRResult(
                raw_text="", confidence=0.0, engine_name=self.name,
                success=False, metadata={"error": f"encode image: {e}"}
            )

        hf_url = f"https://api-inference.huggingface.co/models/{self._model_id}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._hf_token}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self._model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": base64_image}}
                    ]
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }

        try:
            resp = requests.post(hf_url, json=payload, headers=headers, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            elapsed = (time.time() - start) * 1000

            confidence = 0.90 if len(text.strip()) > 50 else 0.82
            return OCRResult(
                raw_text=text.strip(),
                confidence=confidence, language="auto",
                engine_name=self.name, processing_time_ms=elapsed,
                metadata={"model": self._model_id, "mode": "hf_api", "source": "baidu_unlimited_ocr_sota"}
            )
        except Exception as e:
            logger.warning(f"  Unlimited-OCR HF API error: {e}")
            return OCRResult(
                raw_text="", confidence=0.0, engine_name=self.name,
                success=False, metadata={"error": str(e)}
            )
