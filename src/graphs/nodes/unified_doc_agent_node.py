#!/usr/bin/env python3
"""统一文档Agent节点 - V7.0核心创新

学习dots.mocr设计：1个统一Agent + Schema驱动 + 1次API调用搞定结构化提取
"""
import os
import json
import logging
import re
from typing import Dict, Any
from jinja2 import Template
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context

from graphs.state import UnifiedDocAgentInput, UnifiedDocAgentOutput
from utils.llm_client import UnifiedLLMClient
from scenarios import get_scenario_info, get_schema_json_description

logger = logging.getLogger(__name__)


def unified_doc_agent_node(
    state: UnifiedDocAgentInput,
    config: RunnableConfig,
    runtime: Runtime[Context]
) -> UnifiedDocAgentOutput:
    """
    title: 统一文档Agent
    desc: 调用多模态大模型，按Schema一次性提取结构化信息
    integrations: 多模态大语言模型
    """
    ctx = runtime.context

    # 1. 加载配置
    cfg_path = config.get("metadata", {}).get("llm_cfg", "config/unified_agent_llm_cfg.json")
    full_cfg_path = os.path.join(os.getenv("COZE_WORKSPACE_PATH", ""), cfg_path)
    llm_client = UnifiedLLMClient(config_path=full_cfg_path) if os.path.exists(full_cfg_path) else UnifiedLLMClient(
        config_dict={
            "config": {"model": "doubao-seed-2-0-pro-260215", "temperature": 0.05, "max_completion_tokens": 4000},
            "sp": "你是专业文档信息提取引擎。",
            "up": "{{prompt}}"
        }
    )

    # 2. 获取场景信息
    scenario = state.detected_scenario or "general_document"
    scenario_info = get_scenario_info(scenario)
    schema_json = get_schema_json_description(scenario)
    recommended_model = scenario_info.get("recommended_model", "doubao-seed-2-0-pro-260215")

    # 3. 支持运行时模型覆盖
    target_model = state.model_override or recommended_model

    # 4. 构造提取Prompt
    # 优先使用ocr_context作为参考（如果存在），否则仅用图片
    context_section = ""
    if state.ocr_context:
        context_section = f"\n\n## 辅助参考（来自OCR）\n{state.ocr_context[:1000]}"
    # 如果有预提取数据，融合它
    if state.pre_extracted_data:
        context_section += f"\n\n## 已预提取的数据（请补充完善）\n{json.dumps(state.pre_extracted_data, ensure_ascii=False)[:1500]}"

    extract_prompt = f"""请从以下图片中提取{scenario_info.get('display_name', '相关')}信息。{context_section}

## 严格输出要求
1. **仅输出纯JSON**，禁止任何Markdown包裹、解释、注释
2. **字段名必须与Schema完全一致**
3. **缺失字段填空字符串""**，不要省略
4. **数字/日期/金额保持原始格式**
5. **confidence字段输出0-1之间的浮点数**
6. 如果有辅助参考，请结合图片内容修正和完善

## JSON Schema定义
```json
{schema_json}
```

## 输入图片
URL: {state.input_file.url}

## 输出
请直接输出符合上述Schema的JSON对象。"""

    # 5. 调用多模态API
    logger.info(f"统一文档Agent: scenario={scenario}, model={target_model}")
    result = llm_client.call_multimodal(
        prompt=extract_prompt,
        image_url=state.input_file.url,
        model=target_model,
    )

    # 6. 解析JSON结果
    structured_data: Dict[str, Any] = {}
    confidence = 0.0
    parse_success = False

    if result.get("success") and result.get("content"):
        try:
            # 清理可能的Markdown包裹
            content = result["content"].strip()
            # 移除```json ... ```包裹
            if content.startswith("```"):
                content = re.sub(r'^```(?:json)?\s*', '', content)
                content = re.sub(r'\s*```\s*$', '', content)

            structured_data = json.loads(content)
            confidence = float(structured_data.get("confidence", 0.85))
            parse_success = True
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"JSON解析失败: {e}, 原始: {result['content'][:200]}")
            structured_data = {"raw_response": result["content"], "parse_error": str(e)}
            confidence = 0.0
    else:
        structured_data = {"error": "LLM调用失败", "details": result.get("error", "")}
        confidence = 0.0

    logger.info(f"统一Agent完成: scenario={scenario}, success={parse_success}, confidence={confidence}")

    return UnifiedDocAgentOutput(
        structured_data=structured_data,
        scenario_used=scenario,
        model_used=result.get("model_used", target_model),
        confidence=confidence,
        tokens_used=result.get("tokens_used", 0),
        cost=result.get("cost", 0.0),
        parse_success=parse_success,
    )
