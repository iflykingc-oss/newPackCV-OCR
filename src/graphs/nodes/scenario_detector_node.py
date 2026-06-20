#!/usr/bin/env python3
"""场景检测节点 - 自动识别文档类型并匹配最佳Schema"""
import json, os, logging
from typing import Optional
from jinja2 import Template
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context

from graphs.state import ScenarioDetectInput, ScenarioDetectOutput
from utils.scenario_schemas.registry import SchemaRegistry

logger = logging.getLogger(__name__)
registry = SchemaRegistry()


def scenario_detector_node(
    state: ScenarioDetectInput,
    config: RunnableConfig,
    runtime: Runtime[Context],
) -> ScenarioDetectOutput:
    """
    title: 场景智能检测
    desc: 自动识别图片类型（包装/金融票据/银行流水/医药/通用文档），返回最匹配的提取Schema和预处理参数。
    integrations: 大语言模型
    """
    ctx = runtime.context
    ocr_engine_type = getattr(state, "ocr_engine_type", "builtin")
    model_type = getattr(state, "model_type", "extract")
    user_question = getattr(state, "user_question", "")
    target_language = getattr(state, "target_language", "auto")

    # 1. 先用VL模型快速识别文档类型
    detections = {
        "packaging": {"score": 0, "label": "商品包装"},
        "finance_receipt": {"score": 0, "label": "金融票据/发票/收据"},
        "finance_statement": {"score": 0, "label": "银行回单/流水单"},
        "pharmaceutical": {"score": 0, "label": "药品包装/说明书"},
        "contract": {"score": 0, "label": "合同/协议"},
        "id_card": {"score": 0, "label": "证件/身份证/护照"},
        "logistics": {"score": 0, "label": "物流单/快递单"},
        "general_document": {"score": 0, "label": "通用文档"},
    }

    # 从用户提问中提取线索
    if user_question:
        q = user_question.lower()
        if any(k in q for k in ["票据", "发票", "收据", "回单", "流水", "invoice", "receipt", "tax"]):
            for k in ["finance_receipt", "finance_statement"]:
                detections[k]["score"] += 2
        if any(k in q for k in ["药", "药品", "medicine", "drug", "pharma"]):
            detections["pharmaceutical"]["score"] += 2
        if any(k in q for k in ["包装", "食品", "product", "packaging", "标签"]):
            detections["packaging"]["score"] += 2
        if any(k in q for k in ["合同", "协议", "合约", "contract", "agreement"]):
            detections["contract"]["score"] += 2
        if any(k in q for k in ["证件", "身份证", "护照", "驾照", "id card", "passport", "license"]):
            detections["id_card"]["score"] += 2
        if any(k in q for k in ["物流", "快递", "运单", "logistics", "tracking", "shipment"]):
            detections["logistics"]["score"] += 2
        if any(k in q for k in ["文档", "document", "通用", "general", "extract"]):
            detections["general_document"]["score"] += 1

    # 2. 使用VL模型进行多模态场景分类
    try:
        cfg_file = os.path.join(os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects"),
                                config.get("metadata", {}).get("llm_cfg", "config/model_extract_llm_cfg.json"))
        with open(cfg_file) as f:
            _cfg = json.load(f)
        llm_cfg = _cfg.get("config", {})
        model_name = llm_cfg.get("model", "doubao-seed-1.5")

        # 构建VL场景分类Prompt
        classification_prompt = """请分析这张图片，判断它属于以下哪类文档（单选）：
A. 商品包装（食品/饮料/日化/电子产品的包装标签）
B. 金融票据（发票、收据、购物小票）
C. 银行回单/流水单
D. 医药包装（药品包装盒、说明书）
E. 通用文档（其他各类文档、表格、信件）

请只返回一个字母（A/B/C/D/E），不要其他内容。"""

        image_url = getattr(state, "package_image", None)
        image_url_str = image_url.url if image_url and hasattr(image_url, "url") else ""

        # 使用多模态VL模型进行识别
        from coze_coding_dev_sdk import LLMClient
        try:
            from coze_coding_dev_sdk import LLMClient
            from langchain_core.messages import SystemMessage, HumanMessage
            
            client = LLMClient(ctx=ctx)
            vl_messages = [
                SystemMessage(content="你是一个专业的文档图片分类专家，请判断图片类型。"),
                HumanMessage(content=[
                    {"type": "text", "text": classification_prompt},
                    {"type": "image_url", "image_url": {"url": image_url_str}}
                ])
            ]
            vl_response = client.invoke(messages=vl_messages, model=model_name, temperature=0.0, max_tokens=50)
            if isinstance(vl_response, dict):
                choices = vl_response.get("choices", [])
                if choices:
                    vl_text = str(choices[0].get("message", {}).get("content", "")).strip().upper()
            elif isinstance(vl_response, str):
                vl_text = vl_response.strip().upper()
            logger.info(f"[ScenarioDetect] VL分类结果: {vl_text}")
        except Exception as e:
            logger.warning(f"[ScenarioDetect] VL调用失败，使用关键词兜底: {e}")
            vl_text = ""

        # 解析VL响应
        vl_text = ""
        if isinstance(vl_response, str):
            vl_text = vl_response.strip().upper()
        elif isinstance(vl_response, dict):
            choices = vl_response.get("choices", [])
            if choices:
                vl_text = choices[0].get("message", {}).get("content", "").strip().upper()

        scenario_map_vl = {
            "A": "packaging", "B": "finance_receipt",
            "C": "finance_statement", "D": "pharmaceutical",
            "E": "contract", "F": "id_card", "G": "logistics",
            "H": "general_document",
        }
        for key, scenario in scenario_map_vl.items():
            if key in vl_text:
                detections[scenario]["score"] += 5  # VL权重最高
                logger.info(f"VL检测: {scenario}")
                break
    except Exception as e:
        logger.warning(f"VL场景分类失败，使用文本检测: {e}")

    # 3. 选取得分最高的场景
    best_scenario = max(detections, key=lambda k: detections[k]["score"])
    best_label = detections[best_scenario]["label"]
    best_score = detections[best_scenario]["score"]

    # 4. 获取场景Schema和预处理参数
    from utils.scenario_pipeline import ScenarioPipeline
    schema = registry.get(best_scenario)
    preprocess_params = ScenarioPipeline.get_preprocess_params(best_scenario)
    scenario_config = ScenarioPipeline.get_scenario_config(best_scenario)

    # 如果得分很低，降级到通用文档
    if best_score == 0 and best_scenario != "general_document":
        best_scenario = "general_document"
        best_label = "通用文档"
        schema = registry.get("general_document")
        preprocess_params = ScenarioPipeline.get_preprocess_params("general_document")
        scenario_config = ScenarioPipeline.get_scenario_config("general_document")

    logger.info(f"检测结果: {best_label} (scenario={best_scenario}, score={best_score})")

    return ScenarioDetectOutput(
        scenario_type=best_scenario,
        scenario_name=best_label,
        scenario_confidence=min(best_score / 10.0, 1.0),
        scenario_schema_json=json.dumps(scenario_config, ensure_ascii=False),
        preprocess_params=preprocess_params,
        required_fields=[f.name for f in (schema.fields if schema else []) if f.required],
        auto_language=target_language or "auto",
    )