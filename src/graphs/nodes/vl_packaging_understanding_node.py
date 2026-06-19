import os
import json
import re
import logging
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context
from coze_coding_dev_sdk import LLMClient
from langchain_core.messages import SystemMessage, HumanMessage
from jinja2 import Template

from graphs.state import VLPackagingInput, VLPackagingOutput

logger = logging.getLogger(__name__)


def vl_packaging_understanding_node(
    state: VLPackagingInput,
    config: RunnableConfig,
    runtime: Runtime[Context]
) -> VLPackagingOutput:
    """
    title: VL多模态包装理解
    desc: 直接使用多模态大模型(豆包Seed-2.0-Pro)理解包装图片，跳过传统OCR管线，端到端提取所有可见信息
    integrations: 大语言模型
    """
    ctx = runtime.context

    # 读取模型配置
    cfg_file = os.path.join(os.getenv("COZE_WORKSPACE_PATH", ""), config['metadata']['llm_cfg'])
    with open(cfg_file, 'r', encoding='utf-8') as f:
        _cfg = json.load(f)

    llm_config = _cfg.get("config", {})
    sp = _cfg.get("sp", "")
    up = _cfg.get("up", "")

    # 获取图片URL
    image_url = state.package_image.url

    # V5.6 VLM-First模式：如果有OCR文本，作为辅助参考传递给模型
    ocr_reference = state.ocr_reference_text
    vlm_first = getattr(state, 'vlm_primary', True)

    # ====== SmartVL Engine (MiniCPM-o) ======
    smart_vl_used = False
    try:
        engine_cfg_path = os.path.join(
            os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects"),
            "src/config/engine_adapter_cfg.json"
        )
        if os.path.exists(engine_cfg_path) and vlm_first:
            with open(engine_cfg_path) as f:
                engine_cfg = json.load(f)
            minicpm_cfg = engine_cfg.get("vl_engines", {}).get("minicpm_o", {})
            if minicpm_cfg.get("mode") in ("vllm", "hf_api"):
                from utils.vl_engines.minicpm_vl import MiniCPMVLEngine
                mvl = MiniCPMVLEngine(minicpm_cfg)
                if mvl.is_available():
                    logger.info("MiniCPM-o可用，尝试VL理解...")
                    vl_result = mvl.understand(
                        image_url=image_url,
                        prompt="",
                        ocr_hint=ocr_reference or ""
                    )
                    if vl_result.success and len(vl_result.structured_data) > 3:
                        logger.info(f"MiniCPM-o成功: fields={len(vl_result.detected_fields)}")
                        return VLPackagingOutput(
                            vl_success=True,
                            vl_extracted_data=vl_result.structured_data,
                            vl_raw_response=vl_result.raw_response,
                            vl_confidence=vl_result.confidence,
                            engine_used=vl_result.engine_name
                        )
                    else:
                        logger.info(f"MiniCPM-o未达要求, fallback=本地VL")
    except Exception as e:
        logger.warning(f"SmartVL初始化失败: {e}, 降级到本地VL")

    # 读取模型配置
    up_tpl = Template(up)
    image_type = "商品包装"
    if ocr_reference and ocr_reference.strip():
        ocr_note = f"\n\n【辅助OCR参考文本（仅供辅助验证）】\n{ocr_reference[:2000]}\n\n注意：OCR文本仅供参考，请以图片实际观察为准。"
        up_rendered = up_tpl.render({"image_type": image_type}) + ocr_note
    else:
        up_rendered = up_tpl.render({"image_type": image_type})

    logger.info(f"VLM-First模式={'启用' if vlm_first else '禁用'}, OCR参考文本={'有' if ocr_reference else '无'}")

    # 使用LLMClient调用多模态模型
    client = LLMClient(ctx=ctx)

    # 构造多模态消息
    messages = [
        SystemMessage(content=sp),
        HumanMessage(content=[
            {"type": "text", "text": up_rendered},
            {"type": "image_url", "image_url": {"url": image_url}}
        ])
    ]

    try:
        response = client.invoke(
            messages=messages,
            model=llm_config.get("model", "doubao-seed-2-0-pro-260215"),
            temperature=llm_config.get("temperature", 0.05),
            top_p=llm_config.get("top_p", 0.7),
            max_tokens=llm_config.get("max_completion_tokens", 4000),
        )

        # 解析返回文本（兼容str和list类型）
        result_text = ""
        if isinstance(response.content, str):
            result_text = response.content.strip()
        elif isinstance(response.content, list):
            text_parts = []
            for item in response.content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif isinstance(item, str):
                    text_parts.append(item)
            result_text = " ".join(text_parts).strip()
        else:
            result_text = str(response.content)

        # 提取JSON
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            try:
                parsed = json.loads(json_str)
            except json.JSONDecodeError:
                logger.warning("VL理解结果JSON解析失败，使用原始文本")
                parsed = {"raw_extraction": result_text}
        else:
            parsed = {"raw_extraction": result_text}

        # V5.4 商业化统一结构解析：扁平化 category_info 供下游融合使用
        if isinstance(parsed, dict):
            category_info = parsed.pop("category_info", {})
            if not isinstance(category_info, dict):
                category_info = {}
            # 扁平化：把 category_info 的字段提升到顶层（如果顶层没有）
            for k, v in category_info.items():
                if k not in parsed or parsed.get(k) is None:
                    parsed[k] = v
        else:
            category_info = {}

        # V5.6 VLM-First置信度计算增强：基于字段填充率+视觉可验证性
        standard_fields = ["product_type", "brand", "product_name", "specification",
                           "manufacturer", "production_date", "shelf_life"]
        filled = sum(1 for f in standard_fields if parsed.get(f))
        
        # VLM模式下置信度更高（直接视觉观察）
        base_confidence = 0.55 if vlm_first else 0.50
        vl_confidence = min(0.98, base_confidence + filled * 0.055)
        
        # 如果有OCR文本且与VL结果一致，提升置信度
        if ocr_reference and vlm_first:
            vl_confidence = min(0.99, vl_confidence + 0.03)

        logger.info(f"VL理解完成: {filled}/{len(standard_fields)}字段, 置信度={vl_confidence:.3f}")

        return VLPackagingOutput(
            vl_extracted_data=parsed,
            vl_raw_response=result_text,
            vl_confidence=round(vl_confidence, 3),
            vl_success=True,
            vlm_primary_mode=vlm_first,
            ocr_text_used=bool(ocr_reference and ocr_reference.strip())
        )

    except Exception as e:
        logger.error(f"VL模型调用失败: {str(e)}")
        return VLPackagingOutput(
            vl_extracted_data={},
            vl_raw_response=f"VL模型调用失败: {str(e)}",
            vl_confidence=0.0,
            vl_success=False,
            vl_error=str(e)
        )