#!/usr/bin/env python3
"""统一LLM API客户端 - V7.0 核心组件

支持功能:
- 多模态API调用（图片+文本）
- 自动Fallback链（5级降级）
- 智能模型选择
- Token计数和成本估算
- 结构化日志记录
"""
import os
import json
import logging
from typing import Optional, Dict, List, Any
from jinja2 import Template
from langchain_core.messages import SystemMessage, HumanMessage
from coze_coding_dev_sdk import LLMClient as SDKLLMClient

logger = logging.getLogger(__name__)


# 2026年可用模型清单（来自LLM技能）
AVAILABLE_MODELS = {
    "doubao-seed-2-0-pro-260215": {"tier": "premium", "vision": True, "cost_per_1k": 0.008},
    "doubao-seed-2-0-lite-260215": {"tier": "balanced", "vision": True, "cost_per_1k": 0.002},
    "doubao-seed-2-0-mini-260215": {"tier": "fast", "vision": True, "cost_per_1k": 0.0005},
    "doubao-seed-1-8-251228": {"tier": "agent", "vision": True, "cost_per_1k": 0.003},
    "kimi-k2-5-260127": {"tier": "long_context", "vision": True, "cost_per_1k": 0.005},
    "qwen-3-5-plus-260215": {"tier": "vision", "vision": True, "cost_per_1k": 0.004},
    "deepseek-v3-2-251201": {"tier": "text", "vision": False, "cost_per_1k": 0.001},
}

# 默认Fallback链
DEFAULT_FALLBACK_CHAIN = [
    "doubao-seed-2-0-pro-260215",
    "kimi-k2-5-260127",
    "qwen-3-5-plus-260215",
    "doubao-seed-2-0-lite-260215",
    "doubao-seed-2-0-mini-260215",
]


class UnifiedLLMClient:
    """统一LLM API客户端 - 支持多模态+自动Fallback"""

    def __init__(self, config_path: Optional[str] = None, config_dict: Optional[Dict] = None):
        """
        初始化客户端
        :param config_path: LLM配置文件路径（相对COZE_WORKSPACE_PATH）
        :param config_dict: 直接传入配置字典
        """
        if config_path:
            full_path = os.path.join(os.getenv("COZE_WORKSPACE_PATH", ""), config_path)
            with open(full_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        elif config_dict:
            self.config = config_dict
        else:
            raise ValueError("必须提供 config_path 或 config_dict")

        self.model = self.config.get("config", {}).get("model", "doubao-seed-2-0-lite-260215")
        self.temperature = self.config.get("config", {}).get("temperature", 0.05)
        self.max_tokens = self.config.get("config", {}).get("max_completion_tokens", 4000)
        self.sp = self.config.get("sp", "")
        self.up_template = self.config.get("up", "")
        self.tools = self.config.get("tools", [])
        self.fallback_chain = self.config.get("fallback_chain", DEFAULT_FALLBACK_CHAIN)

    def call_multimodal(
        self,
        prompt: str,
        image_url: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        多模态API调用（图片+文本）

        :param prompt: 文本提示词
        :param image_url: 图片URL
        :param model: 临时指定模型（覆盖配置）
        :param temperature: 临时指定温度
        :return: {"content": str, "model_used": str, "tokens": int, "cost": float}
        """
        target_model = model or self.model
        target_temp = temperature if temperature is not None else self.temperature

        # 构造多模态消息
        messages = self._build_multimodal_messages(prompt, image_url)

        # 尝试调用（带Fallback）
        all_models = self._get_model_chain(target_model)
        last_error: Optional[Exception] = None

        for m in all_models:
            try:
                logger.info(f"调用LLM: model={m}, temp={target_temp}")
                client = SDKLLMClient()
                response = client.invoke(
                    model=m,
                    messages=messages,
                    temperature=target_temp,
                    max_completion_tokens=self.max_tokens,
                )

                # 解析响应
                content = self._extract_content(response)
                tokens_used = self._estimate_tokens(prompt, content)
                cost = self._estimate_cost(m, tokens_used)

                logger.info(f"LLM调用成功: model={m}, tokens={tokens_used}, cost=${cost:.4f}")

                return {
                    "content": content,
                    "model_used": m,
                    "tokens_used": tokens_used,
                    "cost": cost,
                    "success": True,
                }
            except Exception as e:
                logger.warning(f"模型 {m} 调用失败: {e}")
                last_error = e
                continue

        # 所有模型都失败
        error_msg = f"所有Fallback模型均失败，最后错误: {last_error}"
        logger.error(error_msg)
        return {
            "content": "",
            "model_used": "",
            "tokens_used": 0,
            "cost": 0.0,
            "success": False,
            "error": error_msg,
        }

    def call_text(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        纯文本API调用
        """
        target_model = model or self.model
        target_temp = temperature if temperature is not None else self.temperature

        messages = []
        if self.sp:
            messages.append(SystemMessage(content=self.sp))
        messages.append(HumanMessage(content=prompt))

        all_models = self._get_model_chain(target_model)
        last_error: Optional[Exception] = None

        for m in all_models:
            try:
                client = SDKLLMClient()
                response = client.invoke(
                    model=m,
                    messages=messages,
                    temperature=target_temp,
                    max_completion_tokens=self.max_tokens,
                )

                content = self._extract_content(response)
                tokens_used = self._estimate_tokens(prompt, content)
                cost = self._estimate_cost(m, tokens_used)

                return {
                    "content": content,
                    "model_used": m,
                    "tokens_used": tokens_used,
                    "cost": cost,
                    "success": True,
                }
            except Exception as e:
                logger.warning(f"模型 {m} 调用失败: {e}")
                last_error = e
                continue

        return {
            "content": "",
            "model_used": "",
            "tokens_used": 0,
            "cost": 0.0,
            "success": False,
            "error": str(last_error) if last_error else "Unknown error",
        }

    def render_prompt(self, **kwargs) -> str:
        """使用Jinja2渲染用户提示词"""
        if not self.up_template:
            return ""
        try:
            tpl = Template(self.up_template)
            return tpl.render(**kwargs)
        except Exception as e:
            logger.warning(f"Prompt渲染失败: {e}, 使用原始模板")
            return self.up_template

    def _build_multimodal_messages(self, prompt: str, image_url: str) -> List:
        """构造多模态消息"""
        messages = []
        if self.sp:
            messages.append(SystemMessage(content=self.sp))
        messages.append(HumanMessage(content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_url}}
        ]))
        return messages

    def _get_model_chain(self, primary: str) -> List[str]:
        """获取模型调用链（primary + fallback）"""
        chain = [primary]
        for m in self.fallback_chain:
            if m != primary and m in AVAILABLE_MODELS:
                chain.append(m)
        return chain

    def _extract_content(self, response: Any) -> str:
        """从LLM响应中提取文本内容"""
        if isinstance(response, str):
            return response
        if hasattr(response, 'content'):
            return str(response.content)
        if isinstance(response, dict):
            return response.get("content", str(response))
        return str(response)

    def _estimate_tokens(self, prompt: str, response: str) -> int:
        """估算Token数量（中英文混合）"""
        # 简单估算：1个中文字符≈1.5 tokens，1个英文单词≈1.3 tokens
        chinese_chars = sum(1 for c in prompt + response if '\u4e00' <= c <= '\u9fff')
        other_chars = len(prompt) + len(response) - chinese_chars
        return int(chinese_chars * 1.5 + other_chars * 0.3)

    def _estimate_cost(self, model: str, tokens: int) -> float:
        """估算API调用成本（美元）"""
        if model not in AVAILABLE_MODELS:
            return 0.0
        cost_per_1k = AVAILABLE_MODELS[model].get("cost_per_1k", 0.0)
        return round(tokens / 1000 * cost_per_1k, 6)


def quick_llm_call(prompt: str, image_url: str = "", model: str = "doubao-seed-2-0-lite-260215") -> str:
    """快速LLM调用 - 简化版本"""
    config = {
        "config": {
            "model": model,
            "temperature": 0.05,
            "max_completion_tokens": 2000,
        },
        "sp": "你是专业助手。",
        "up": prompt,
    }
    client = UnifiedLLMClient(config_dict=config)
    if image_url:
        result = client.call_multimodal(prompt, image_url)
    else:
        result = client.call_text(prompt)
    return result.get("content", "")
