# -*- coding: utf-8 -*-
"""
多平台OCR包装识别系统 - 主图编排 V6.1
整合V6.0(MinerU文档引擎/输入路由/条码印章/smart_postprocess) + V6.1(OCR融合/LLM纠错/表格检测/VLM辅助)

图片路径: route → scenario_detector → preprocess → enhance → curvature → quality_router
  → [OCR+VL并行] → multi_channel_fusion(含条码+印章检测) → smart_postprocess → QA(条件) → result → feishu → END
文档路径: route → input_router → document_parse → result → feishu → END
"""

from typing import Optional
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context
from graphs.state import (
    GlobalState,
    GraphInput,
    GraphOutput,
    RouteProcessingInput,
    RouteProcessingOutput,
    InputTypeRouteInput,
    InputTypeRouteOutput,
    QaConditionalInput
)

# 核心节点
from graphs.nodes.image_preprocess_node import image_preprocess_node
from graphs.nodes.ocr_recognize_node import ocr_recognize_node
from graphs.nodes.model_extract_node import model_extract_node
from graphs.nodes.correct_text_node import correct_text_node
from graphs.nodes.qa_answer_node import qa_answer_node
from graphs.nodes.result_output_node import result_output_node
from graphs.nodes.batch_process_node import batch_process_node
from graphs.nodes.feishu_notify_node import feishu_notify_node
# VL多模态
from graphs.nodes.vl_packaging_understanding_node import vl_packaging_understanding_node
from graphs.nodes.knowledge_inference_node import knowledge_inference_node
# V5.3 深度优化
from graphs.nodes.multi_channel_fusion_node import multi_channel_fusion_node
from graphs.nodes.category_template_node import category_template_node
from graphs.nodes.call_audit_node import call_audit_node
from graphs.nodes.multi_language_ocr_node import multi_language_ocr_node
# V5.5 图像质量路由
from graphs.nodes.image_quality_router_node import image_quality_router_node
# V5.6 能力提升
from graphs.nodes.image_quality_enhance_node import image_quality_enhance_node
from graphs.nodes.text_curvature_correct_node import text_curvature_correct_node
from graphs.nodes.multi_language_ocr_enhanced_node import multi_language_ocr_enhanced_node
# V5.8 场景检测
from graphs.nodes.scenario_detector_node import scenario_detector_node
# V6.0 输入路由 + 文档解析 + 智能后处理
from graphs.nodes.input_router_node import input_router_node
from graphs.nodes.document_parse_node import document_parse_node
from graphs.nodes.smart_postprocess_node import smart_postprocess_node


def route_processing_node(state: RouteProcessingInput, config: RunnableConfig, runtime: Runtime[Context]) -> RouteProcessingOutput:
    """
    title: 路由处理模式
    desc: 根据输入决定是单图处理还是批量处理
    """
    ctx = runtime.context
    images = state.images
    if images and isinstance(images, list) and len(images) > 1:
        return RouteProcessingOutput(processing_mode="batch")
    else:
        return RouteProcessingOutput(processing_mode="single")


def route_processing_condition(state: GlobalState) -> str:
    """
    title: 路由处理模式（条件分支）
    desc: 根据processing_mode字段决定处理流程
    """
    processing_mode = getattr(state, 'processing_mode', 'single')
    if processing_mode == "batch":
        return "批量处理"
    else:
        return "单图处理"


def input_type_condition(state: InputTypeRouteInput) -> str:
    """
    title: 输入类型路由
    desc: 根据文件类型判断走图片管线还是文档管线
    """
    input_type = getattr(state, 'input_type', 'image')
    if input_type == "document":
        return "文档解析"
    else:
        return "图片处理"


def qa_conditional(state: QaConditionalInput) -> str:
    """
    title: 是否需要QA回答
    desc: 检查user_question是否非空，非空则触发QA节点
    """
    user_question = getattr(state, 'user_question', '') or ''
    if user_question.strip():
        return "需要回答"
    else:
        return "直接输出"


# 创建状态图
builder = StateGraph(GlobalState, input_schema=GraphInput, output_schema=GraphOutput)

# ===== 核心节点 =====
builder.add_node("route_processing", route_processing_node)
builder.add_node("batch_process", batch_process_node)
builder.add_node("image_preprocess", image_preprocess_node)
builder.add_node("ocr_recognize", ocr_recognize_node)
builder.add_node("model_extract", model_extract_node,
                 metadata={"type": "agent", "llm_cfg": "config/model_extract_llm_cfg.json"})
builder.add_node("correct_text", correct_text_node,
                 metadata={"type": "agent", "llm_cfg": "config/correct_text_llm_cfg.json"})
builder.add_node("qa_answer", qa_answer_node,
                 metadata={"type": "agent", "llm_cfg": "config/qa_answer_llm_cfg.json"})
builder.add_node("result_output", result_output_node)
builder.add_node("feishu_notify", feishu_notify_node)

# ===== VL多模态 =====
builder.add_node("vl_packaging_understanding", vl_packaging_understanding_node,
                 metadata={"type": "agent", "llm_cfg": "config/vl_packaging_llm_cfg.json"})

# ===== 多通道融合(内嵌条码+印章检测) / 调用审计 / 多语言OCR =====
builder.add_node("multi_channel_fusion", multi_channel_fusion_node)
builder.add_node("call_audit", call_audit_node)
builder.add_node("multi_language_ocr", multi_language_ocr_node)

# ===== 图像质量路由 + V5.6增强管线 =====
builder.add_node("image_quality_router", image_quality_router_node)
builder.add_node("image_quality_enhance", image_quality_enhance_node)
builder.add_node("text_curvature_correct", text_curvature_correct_node)
builder.add_node("multi_language_ocr_enhanced", multi_language_ocr_enhanced_node)

# ===== V5.8 场景自动检测 =====
builder.add_node("scenario_detector", scenario_detector_node,
                 metadata={"type": "agent", "llm_cfg": "config/finance_extract_llm_cfg.json"})

# ===== V6.0 输入路由 + 文档解析 + 智能后处理 =====
builder.add_node("input_router", input_router_node)
builder.add_node("document_parse", document_parse_node,
                 metadata={"type": "agent", "llm_cfg": "config/document_extract_llm_cfg.json"})
builder.add_node("smart_postprocess", smart_postprocess_node,
                 metadata={"type": "agent", "llm_cfg": "config/knowledge_inference_llm_cfg.json"})

# ===== 设置入口点 =====
builder.set_entry_point("route_processing")

# ===== 主路由：批量 vs 单图 =====
builder.add_conditional_edges(
    source="route_processing",
    path=route_processing_condition,
    path_map={
        "批量处理": "batch_process",
        "单图处理": "input_router"
    }
)

# ===== V6.0 输入类型路由：图片 vs 文档 =====
builder.add_conditional_edges(
    source="input_router",
    path=input_type_condition,
    path_map={
        "图片处理": "scenario_detector",
        "文档解析": "document_parse"
    }
)

# ===== 图片处理管线 =====
# 场景检测 → 预处理 → 质量增强 → 弯曲校正 → 质量路由
builder.add_edge("scenario_detector", "image_preprocess")
builder.add_edge("image_preprocess", "image_quality_enhance")
builder.add_edge("image_quality_enhance", "text_curvature_correct")
builder.add_edge("text_curvature_correct", "image_quality_router")

# 4路并行：OCR + VL (主提取通道)
builder.add_edge("image_quality_router", "ocr_recognize")
builder.add_edge("image_quality_router", "vl_packaging_understanding")

# OCR通道 → 纠错 → 结构化提取
builder.add_edge("ocr_recognize", "correct_text")
builder.add_edge("correct_text", "model_extract")

# 2路并行汇聚到多通道融合(含条码+印章内嵌检测)
builder.add_edge(["model_extract", "vl_packaging_understanding"], "multi_channel_fusion")

# V6.0 智能后处理(知识推理+品类模板合并，单次LLM)
builder.add_edge("multi_channel_fusion", "smart_postprocess")

# V6.0 QA条件触发：有用户提问才走QA，否则直出结果
builder.add_conditional_edges(
    source="smart_postprocess",
    path=qa_conditional,
    path_map={
        "需要回答": "qa_answer",
        "直接输出": "result_output"
    }
)
builder.add_edge("qa_answer", "result_output")

# 结果输出 → 审计 → 飞书 → END
builder.add_edge("result_output", "call_audit")
builder.add_edge("call_audit", "feishu_notify")
builder.add_edge("feishu_notify", END)

# ===== 文档处理管线 =====
# document_parse → 直接出结果(MinerU已做结构化)
builder.add_edge("document_parse", "result_output")

# ===== 批量处理 =====
builder.add_edge("batch_process", END)

# 编译图
main_graph = builder.compile()
