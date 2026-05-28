# -*- coding: utf-8 -*-
"""
模型结构化提取节点
使用大语言模型从OCR文本中提取结构化信息，支持规则引擎降级
"""

import os
import re
import json
import time
from jinja2 import Template
from typing import Dict, Any, List
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context
from graphs.state import ModelExtractInput, ModelExtractOutput
from langchain_core.messages import SystemMessage, HumanMessage
from coze_coding_dev_sdk import LLMClient


def rule_based_extract(ocr_text: str, template_fields: List[str]) -> Dict[str, Any]:
    """
    基于规则的结构化提取，不依赖LLM
    使用正则表达式从OCR文本中提取关键字段
    """
    result: Dict[str, Any] = {}

    rules = {
        "brand": [
            r"(?:品牌|商标|生产商)[：:]\s*([\u4e00-\u9fa5A-Za-z0-9]+)",
            r"([\u4e00-\u9fa5]{2,8}(?:牌|集团|股份|有限|公司|厂))",
            r"(?:金龙鱼|海天|康师傅|统一|蒙牛|伊利|娃哈哈|王老吉|加多宝|百事|可口可乐|雀巢|达利园|三只松鼠|良品铺子)",
        ],
        "product_name": [
            r"(?:产品名称|品名|名称)[：:]\s*([\u4e00-\u9fa5A-Za-z0-9/（()）]+)",
            r"([\u4e00-\u9fa5]*(?:油|奶|饮料|酒|米|面|茶|糖|粉|酱|醋|调料|食品|用品))",
        ],
        "specification": [
            r"(?:净含量|规格|容量|体积|重量)[：:]*\s*([\d.]+\s*(?:L|ml|g|kg|毫升|升|克|千克))",
            r"([\d.]+\s*(?:L|ml|g|kg|毫升|升|克|千克))",
        ],
        "production_date": [
            r"(?:生产日期|生产|日期|制造日期)[：:]*\s*(\d{4}[-/年.]?\d{1,2}[-/月.]?\d{1,2})",
            r"(\d{4}[-/年]\d{1,2}[-/月]\d{1,2})",
            r"(\d{4}[-/.]\d{2}[-/.]\d{2})",
        ],
        "shelf_life": [
            r"(?:保质期|有效期|有效期限|保质)[：:]*\s*(\d{1,3}\s*(?:天|日|个月|年))",
            r"(?:保质期至|有效期至|有效期到)[：:]*\s*(\d{4}[-/年.]?\d{1,2}[-/月.]?\d{1,2})",
        ],
        "manufacturer": [
            r"(?:制造商|生产商|生产厂家|厂家|委托方|出品)[：:]\s*([\u4e00-\u9fa5A-Za-z0-9（()）]+)",
        ],
        "ingredients": [
            r"(?:配料|成分|原料|材质)[：:]\s*(.+?)(?:\n|$)",
        ],
        "standard": [
            r"(?:执行标准|标准号|产品标准)[：:]\s*([A-Z0-9/.\-]+)",
        ],
        "batch_number": [
            r"(?:批号|批次|生产批号)[：:]\s*([A-Za-z0-9\-]+)",
        ],
        "license_number": [
            r"(?:生产许可证|许可证|SC编号|食品生产许可证)[：:]\s*([A-Za-z0-9\-]+)",
        ],
    }

    for field, patterns in rules.items():
        for pattern in patterns:
            match = re.search(pattern, ocr_text)
            if match:
                try:
                    value = match.group(1).strip() if match.lastindex and match.lastindex >= 1 else match.group(0).strip()
                    if value:
                        result[field] = value
                        break
                except IndexError:
                    result[field] = match.group(0).strip()
                    break

    if template_fields:
        for field in template_fields:
            if field not in result:
                result[field] = "N/A"
    else:
        default_fields = ["brand", "product_name", "specification", "production_date", "shelf_life", "manufacturer"]
        for field in default_fields:
            if field not in result:
                result[field] = "N/A"
        for field in rules:
            if field not in default_fields and field in result:
                result[field] = result[field]

    return result


def model_extract_node(
    state: ModelExtractInput,
    config: RunnableConfig,
    runtime: Runtime[Context]
) -> ModelExtractOutput:
    """
    title: 结构化信息提取
    desc: 使用大语言模型从OCR识别文本中提取结构化信息（如品牌、规格、生产日期等）
    integrations: 大语言模型
    """
    ctx = runtime.context

    # 预先获取ocr_text（避免在except块中使用未定义变量）
    ocr_text = state.ocr_text or state.raw_text or state.ocr_raw_result or ""

    try:
        # 加载模型配置
        cfg_file = os.path.join(os.getenv("COZE_WORKSPACE_PATH"), config['metadata']['llm_cfg'])
        with open(cfg_file, 'r', encoding='utf-8') as f:
            _cfg = json.load(f)

        llm_config = _cfg.get("config", {})
        sp = _cfg.get("sp", "")
        up_tpl = Template(_cfg.get("up", ""))

        # 使用自定义提示词或默认提示词
        user_prompt = state.custom_prompt if state.custom_prompt else up_tpl.render({
            "ocr_text": ocr_text,
            "fields": json.dumps(state.template_fields, ensure_ascii=False) if state.template_fields else ""
        })

        # 构造消息
        messages = [
            SystemMessage(content=sp),
            HumanMessage(content=user_prompt)
        ]

        # 初始化并调用模型
        client = LLMClient(ctx=ctx)

        model_params = {
            "model": llm_config.get("model", state.model_name),
            "temperature": llm_config.get("temperature", 0.1),
            "max_tokens": llm_config.get("max_completion_tokens", 2000),
        }

        response = client.invoke(
            messages=messages,
            **model_params
        )

        # 解析响应 - 安全获取文本内容
        result_text = ""
        if response and hasattr(response, 'content'):
            content = response.content
            if isinstance(content, str):
                result_text = content
            elif isinstance(content, list):
                if content and isinstance(content[0], str):
                    result_text = " ".join(content)
                else:
                    text_parts = [item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"]
                    result_text = " ".join(text_parts).strip()
            else:
                result_text = str(content)

        # 尝试解析JSON
        structured_data: Dict[str, Any] = {}
        confidence = 0.0
        missing_fields: List[str] = []

        try:
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                structured_data = json.loads(json_match.group())
            else:
                structured_data = json.loads(result_text)

            if state.template_fields:
                filled_fields = [f for f in state.template_fields if f in structured_data and structured_data[f]]
                missing_fields = [f for f in state.template_fields if f not in filled_fields]
                confidence = len(filled_fields) / len(state.template_fields) if state.template_fields else 1.0
            else:
                confidence = 0.8

        except json.JSONDecodeError as e:
            print(f"JSON解析失败，尝试文本提取: {str(e)}")
            structured_data = {"raw_extract": result_text}
            confidence = 0.5

        print(f"结构化提取完成，置信度: {confidence:.2f}")

        return ModelExtractOutput(
            structured_data=structured_data,
            confidence=confidence,
            missing_fields=missing_fields
        )

    except Exception as e:
        print(f"模型结构化提取失败: {str(e)}，降级到规则引擎")
        # 降级到规则引擎 - ocr_text已在函数开头安全获取
        rule_data = rule_based_extract(ocr_text, state.template_fields or [])
        filled_count = sum(1 for v in rule_data.values() if v != "N/A")
        rule_confidence = filled_count / len(rule_data) if rule_data else 0.3

        print(f"规则引擎提取完成: {json.dumps(rule_data, ensure_ascii=False)}, 置信度: {rule_confidence:.2f}")
        return ModelExtractOutput(
            structured_data=rule_data,
            confidence=rule_confidence,
            missing_fields=[f for f in rule_data if rule_data[f] == "N/A"]
        )
