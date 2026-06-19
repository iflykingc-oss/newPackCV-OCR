"""MiniCPM-o 适配器

支持两种模式：
1. vLLM/OpenAI兼容API
2. HuggingFace Inference API
"""

import os, time, logging, base64, json, re
from typing import Optional, Dict, Any

from utils.vl_engines.base import BaseVLEngine, VLResult

logger = logging.getLogger(__name__)


class MiniCPMVLEngine(BaseVLEngine):
    """MiniCPM-o 8B VL引擎适配器"""

    MODE_VLLM = "vllm"
    MODE_HF_API = "hf_api"
    MODE_DISABLED = "disabled"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or {}
        self._mode = self._config.get("mode", self.MODE_DISABLED)
        self._endpoint = self._config.get("endpoint", "")
        self._hf_token = self._config.get("hf_token", os.getenv("HF_TOKEN", ""))
        # MiniCPM-o 最新版本 4.5
        self._model_id = self._config.get("model_id", "openbmb/MiniCPM-o-4_5")
        self._max_tokens = self._config.get("max_tokens", 4096)
        self._temperature = self._config.get("temperature", 0.1)

    @property
    def name(self) -> str:
        return f"MiniCPM-o({self._mode})"

    def is_available(self) -> bool:
        if self._mode == self.MODE_VLLM:
            return bool(self._endpoint)
        elif self._mode == self.MODE_HF_API:
            return bool(self._hf_token)
        return False

    def understand(self, image_url: str, prompt: str = "",
                   ocr_hint: str = "", options: Optional[Dict[str, Any]] = None) -> VLResult:
        start = time.time()
        opts = {**(options or {})}
        opts["prompt"] = prompt
        opts["ocr_hint"] = ocr_hint

        if self._mode == self.MODE_VLLM:
            return self._call_vllm(image_url, opts)
        elif self._mode == self.MODE_HF_API:
            return self._call_hf_api(image_url, opts)
        else:
            return VLResult(
                structured_data={}, raw_response="", confidence=0.0,
                engine_name=self.name, success=False,
                metadata={"error": f"MiniCPM-o mode={self._mode} not available"}
            )

    def _build_prompt(self, opts: Dict[str, Any]) -> str:
        """构造VLM理解的提示词"""
        prompt = opts.get("prompt", "").strip()
        ocr_hint = opts.get("ocr_hint", "").strip()

        if not prompt:
            prompt = """请仔细观察这张商品包装图片，提取其中的结构化信息。
严格按照以下JSON格式返回（可直接JSON.parse解析）：

{
  "product_type": "食品/饮料/日化清洁/个人护理/药品/电子产品/其他",
  "brand": "品牌名称",
  "product_name": "产品全称",
  "specification": "规格/净含量",
  "manufacturer": "生产商",
  "production_date": "生产日期 YYYY-MM-DD",
  "shelf_life": "保质期",
  "batch_number": "批号",
  "warnings": ["注意事项1", "注意事项2"],
  "category_info": {
    "ingredients": ["配料1", "配料2"],
    "nutrition_info": ["营养成分1"],
    "usage_method": "使用方法",
    "storage_condition": "贮存条件"
  },
  "ext_info": ["其他信息1"]
}

约束：
1. 仅提取图片中实际可见的信息，不要编造
2. 无法看到的字段请设为null或空数组
3. 品类必须从指定的7类中选择

请直接返回JSON，不要添加任何其他文字。"""

        if ocr_hint:
            prompt += f"\n\n作为参考，OCR识别到的部分文字如下（可能有误）：\n{ocr_hint[:1000]}"
            prompt += "\n\n请以你的视觉理解为主，OCR文本仅供参考。"

        return prompt

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """从模型响应中提取JSON"""
        # 尝试直接解析
        text = text.strip()
        # 去掉可能的markdown代码块
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # 尝试查找JSON对象
        m = re.search(r'\{[\s\S]*\}', text)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
        return {"product_type": "其他", "ext_info": []}

    # ── vLLM API ────────────────────────────────────────────────

    def _call_vllm(self, image_url: str, opts: Dict[str, Any]) -> VLResult:
        import requests
        max_tokens = opts.get("max_tokens", self._max_tokens)
        temperature = opts.get("temperature", self._temperature)
        timeout = opts.get("timeout", 120)
        prompt = self._build_prompt(opts)

        payload = {
            "model": self._model_id,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": prompt}
                ]
            }],
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
            raw_text = data["choices"][0]["message"]["content"]
            sd = self._parse_json_response(raw_text)
            elapsed = (time.time() - start) * 1000
            fields = [k for k, v in sd.items() if v is not None and v != "" and v != [] and v != {}]
            return VLResult(
                structured_data=sd, raw_response=raw_text,
                confidence=0.9, engine_name=self.name,
                processing_time_ms=elapsed, detected_fields=fields,
                metadata={"model": self._model_id, "mode": "vllm"}
            )
        except Exception as e:
            logger.warning(f"MiniCPM-o vLLM error: {e}")
            return VLResult(
                structured_data={}, raw_response="", confidence=0.0,
                engine_name=self.name, success=False,
                metadata={"error": str(e)}
            )

    # ── HF Inference API ────────────────────────────────────────

    def _call_hf_api(self, image_url: str, opts: Dict[str, Any]) -> VLResult:
        import requests
        max_tokens = opts.get("max_tokens", self._max_tokens)
        timeout = opts.get("timeout", 180)
        prompt = self._build_prompt(opts)

        # 下载图片转base64
        try:
            resp_img = requests.get(image_url, timeout=30)
            resp_img.raise_for_status()
            b64 = base64.b64encode(resp_img.content).decode("utf-8")
            data_uri = f"data:image/jpeg;base64,{b64}"
        except Exception as e:
            return VLResult(
                structured_data={}, raw_response="", confidence=0.0,
                engine_name=self.name, success=False,
                metadata={"error": f"download: {e}"}
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
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": prompt}
                ]
            }],
            "max_tokens": max_tokens,
            "temperature": 0.1,
        }

        try:
            resp = requests.post(hf_url, json=payload, headers=headers, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            raw_text = data["choices"][0]["message"]["content"]
            sd = self._parse_json_response(raw_text)
            elapsed = (time.time() - start) * 1000
            fields = [k for k, v in sd.items() if v is not None and v != "" and v != [] and v != {}]
            return VLResult(
                structured_data=sd, raw_response=raw_text,
                confidence=0.85, engine_name=self.name,
                processing_time_ms=elapsed, detected_fields=fields,
                metadata={"model": self._model_id, "mode": "hf_api"}
            )
        except Exception as e:
            logger.warning(f"MiniCPM-o HF API error: {e}")
            return VLResult(
                structured_data={}, raw_response="", confidence=0.0,
                engine_name=self.name, success=False,
                metadata={"error": str(e)}
            )