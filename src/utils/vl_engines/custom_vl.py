"""通用自定义VL引擎适配器

支持任何兼容OpenAI API的视觉-语言模型端点（vLLM / Ollama / TGI / 自部署等）。
用户可在配置文件中定义任意多个自定义VL模型。

配置示例:
  "custom_engines": [
    {
      "name": "my-vllm-vl",
      "endpoint": "https://my-server/v1/chat/completions",
      "model": "Qwen/Qwen2-VL-7B-Instruct",
      "api_key": "sk-xxx",
      "priority": 0,
      "prompt": "请根据图片详细描述产品包装信息",
      "json_mode": true
    }
  ]

优先级规则：priority值越小优先级越高（0 > 1 > 2）
json_mode: true时强制模型返回JSON格式
"""

import json, os, time, logging, base64, re
from typing import Optional, Dict, Any
from urllib.request import Request, urlopen
from urllib.error import URLError

from utils.vl_engines.base import BaseVLEngine, VLResult

logger = logging.getLogger(__name__)

DEFAULT_VL_PROMPT = (
    "请仔细观察这张产品包装图片，提取所有可见的结构化信息。"
    "包括：产品类型、品牌、产品名称、规格、生产商、生产日期、保质期、批号、"
    "配料表/成分表、使用方法、注意事项、贮存条件等。"
    "以JSON格式返回，字段名为英文小写，值为提取到的信息。"
)

DEFAULT_JSON_PROMPT = (
    "请仔细观察这张产品包装图片，提取结构化信息。"
    "仅返回纯JSON，不要包含任何markdown标记或解释。"
    "{\n"
    '  "product_type": "品类名称",\n'
    '  "brand": "品牌",\n'
    '  "product_name": "产品名称",\n'
    '  "specification": "规格/净含量",\n'
    '  "manufacturer": "生产商",\n'
    '  "production_date": "YYYY-MM-DD",\n'
    '  "shelf_life": "保质期",\n'
    '  "batch_number": "批号",\n'
    '  "ingredients": ["配料1", "配料2"],\n'
    '  "usage_method": "使用方法",\n'
    '  "warnings": "注意事项",\n'
    '  "storage_condition": "贮存条件"\n'
    "}"
)


class CustomVLEngine(BaseVLEngine):
    """通用自定义VL引擎适配器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or {}
        self._name = self._config.get("name", "custom-vl")
        self._endpoint = self._config.get("endpoint", "")
        self._api_key = self._config.get("api_key", os.getenv("CUSTOM_MODEL_API_KEY", ""))
        self._model = self._config.get("model", "")
        self._prompt = self._config.get("prompt", DEFAULT_VL_PROMPT)
        self._json_mode = self._config.get("json_mode", False)
        self._priority = self._config.get("priority", 99)
        self._max_tokens = self._config.get("max_tokens", 4096)
        self._temperature = self._config.get("temperature", 0.1)
        self._last_fail_ts = 0.0
        self._retry_count = 0

    @property
    def name(self) -> str:
        return f"CustomVL[{self._name}]"

    @property
    def priority(self) -> int:
        return self._priority

    def is_available(self) -> bool:
        if not self._endpoint or not self._model:
            return False
        if self._retry_count >= 3:
            cooldown = time.time() - self._last_fail_ts
            if cooldown < 600:
                return False
            self._retry_count = 0
        return True

    def _encode_image(self, image_url: str) -> str:
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

    def _parse_structured_data(self, raw_text: str) -> Dict[str, Any]:
        """从模型输出中解析结构化数据"""
        text = raw_text.strip()
        # 尝试提取JSON块
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        # 尝试提取JSON数组
        list_match = re.search(r'\[[\s\S]*\]', text)
        if list_match:
            try:
                return {"items": json.loads(list_match.group())}
            except json.JSONDecodeError:
                pass
        return {"raw_description": text}

    def understand(self, image_url: str, prompt: str = "",
                   ocr_hint: str = "", options: Optional[Dict[str, Any]] = None) -> VLResult:
        start = time.time()
        opts = {**(options or {})}

        try:
            base64_image = self._encode_image(image_url)
            use_prompt = opts.get("prompt", prompt) or self._prompt
            if self._json_mode:
                use_prompt = DEFAULT_JSON_PROMPT
            if ocr_hint:
                use_prompt += f"\n\nOCR参考文本（可能不准确）:\n{ocr_hint[:1000]}"

            payload = {
                "model": self._model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": use_prompt},
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
            with urlopen(req, timeout=180) as resp:
                response_data = json.loads(resp.read().decode())

            raw_response = ""
            choices = response_data.get("choices", [])
            if choices:
                raw_response = choices[0].get("message", {}).get("content", "")
                if not raw_response:
                    raw_response = choices[0].get("text", "")

            elapsed = (time.time() - start) * 1000

            if raw_response.strip():
                structured = self._parse_structured_data(raw_response)
                detected_fields = [k for k, v in structured.items() if v and v != "null"]
                confidence = min(0.95, 0.5 + len(detected_fields) * 0.05)

                self._retry_count = 0
                logger.info(f"  ✓ {self._name}: {len(detected_fields)} fields in {elapsed:.0f}ms")
                return VLResult(
                    structured_data=structured,
                    raw_response=raw_response.strip(),
                    confidence=confidence,
                    detected_fields=detected_fields,
                    engine_name=self.name,
                    processing_time_ms=elapsed,
                    success=True,
                    metadata={"model": self._model, "endpoint": self._endpoint}
                )

            self._retry_count += 1
            self._last_fail_ts = time.time()
            return VLResult(
                structured_data={}, raw_response="", confidence=0.0,
                engine_name=self.name, success=False,
                metadata={"error": "empty response"}
            )

        except Exception as e:
            self._retry_count += 1
            self._last_fail_ts = time.time()
            elapsed = (time.time() - start) * 1000
            logger.warning(f"  ✗ {self._name}: {e}")
            return VLResult(
                structured_data={}, raw_response="", confidence=0.0,
                engine_name=self.name, success=False,
                metadata={"error": str(e)}
            )