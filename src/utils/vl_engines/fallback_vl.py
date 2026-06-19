"""降级VL引擎 - 包装现有管线中的VL理解能力"""

import time, logging, json, re, os
from typing import Optional, Dict, Any
from jinja2 import Template

from utils.vl_engines.base import BaseVLEngine, VLResult

logger = logging.getLogger(__name__)


class FallbackVLEngine(BaseVLEngine):
    """降级VL引擎 - 使用原有VL管线"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or {}
        self._llm_config_path = self._config.get(
            "llm_config", "config/vl_packaging_llm_cfg.json"
        )
        self._llm_client = None

    @property
    def name(self) -> str:
        return "FallbackVL"

    def is_available(self) -> bool:
        return True  # 始终可用

    def _get_llm_client(self):
        if self._llm_client is None:
            from coze_coding_dev_sdk import LLMClient
            self._llm_client = LLMClient()
        return self._llm_client

    def _load_config(self) -> Dict[str, Any]:
        path = os.path.join(os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects"), self._llm_config_path)
        with open(path, "r") as f:
            return json.load(f)

    def understand(self, image_url: str, prompt: str = "",
                   ocr_hint: str = "", options: Optional[Dict[str, Any]] = None) -> VLResult:
        start = time.time()
        opts = {**(options or {})}

        try:
            cfg = self._load_config()
            sp = cfg.get("sp", "")
            up_tpl = cfg.get("up", "")

            # 构造提示词
            if ocr_hint:
                up = Template(up_tpl).render({"ocr_text": ocr_hint})
            else:
                up = prompt if prompt else "请仔细观察这张商品包装图片，提取结构化信息。"

            # 多模态大模型：消息格式
            from langchain_core.messages import HumanMessage
            msg = HumanMessage(content=[
                {"type": "text", "text": f"{sp}\n\n{up}"},
                {"type": "image_url", "image_url": {"url": image_url}}
            ])

            client = self._get_llm_client()
            llm_cfg = cfg.get("config", {})
            resp = client.invoke(
                messages=[msg],
                model=llm_cfg.get("model", "doubao-seed-1.5"),
                temperature=0.0,
                max_tokens=4096
            )

            content = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
            elapsed = (time.time() - start) * 1000

            # 解析JSON
            sd = self._parse_json(content)
            fields = [k for k, v in sd.items() if v is not None and v != "" and v != [] and v != {}]

            return VLResult(
                structured_data=sd, raw_response=content,
                confidence=0.75, engine_name=self.name,
                processing_time_ms=elapsed, detected_fields=fields,
                metadata={"llm_model": llm_cfg.get("model", "")}
            )
        except Exception as e:
            logger.warning(f"FallbackVL error: {e}")
            return VLResult(
                structured_data={}, raw_response="", confidence=0.0,
                engine_name=self.name, success=False,
                metadata={"error": str(e)}
            )

    def _parse_json(self, text: str) -> Dict[str, Any]:
        text = text.strip()
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        m = re.search(r'\{[\s\S]*\}', text)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
        return {"product_type": "其他", "ext_info": []}