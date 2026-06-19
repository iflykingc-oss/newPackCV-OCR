# -*- coding: utf-8 -*-
"""
多平台OCR包装识别系统 - 主图编排 V5.3
深度优化版：集成多通道融合、品类模板、多语言OCR、调用审计
串行DAG: route → image_preprocess → ocr_recognize → correct_text → model_extract
  → multi_channel_fusion (融合VL) → knowledge_inference → category_template
  → qa_answer → result_output → call_audit → feishu_notify
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
    ResultOutputInput,
    RouteProcessingInput,
    RouteProcessingOutput
)

# 导入所有节点
from graphs.nodes.image_preprocess_node import image_preprocess_node
from graphs.nodes.ocr_recognize_node import ocr_recognize_node
from graphs.nodes.model_extract_node import model_extract_node
from graphs.nodes.correct_text_node import correct_text_node
from graphs.nodes.qa_answer_node import qa_answer_node
from graphs.nodes.result_output_node import result_output_node
from graphs.nodes.batch_process_node import batch_process_node
from graphs.nodes.feishu_notify_node import feishu_notify_node
from graphs.nodes.vl_packaging_understanding_node import vl_packaging_understanding_node
from graphs.nodes.knowledge_inference_node import knowledge_inference_node
# V5.3 深度优化新节点
from graphs.nodes.multi_channel_fusion_node import multi_channel_fusion_node
from graphs.nodes.category_template_node import category_template_node
from graphs.nodes.call_audit_node import call_audit_node
from graphs.nodes.multi_language_ocr_node import multi_language_ocr_node


def route_processing_mode(state: GlobalState) -> str:
    """
    title: 路由处理模式
    desc: 根据输入决定是单图处理还是批量处理：如果images字段存在且有多个图片，则进入批量处理
    """
    images = getattr(state, 'images', None)
    if images and isinstance(images, list) and len(images) > 1:
        return "批量处理"
    else:
        return "单图处理"


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


# 创建状态图
builder = StateGraph(GlobalState, input_schema=GraphInput, output_schema=GraphOutput)

# 添加节点
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

# ===== 深度优化方向⑥-⑦：VL多模态 + 知识图谱推理 =====
builder.add_node("vl_packaging_understanding", vl_packaging_understanding_node,
                 metadata={"type": "agent", "llm_cfg": "config/vl_packaging_llm_cfg.json"})
builder.add_node("knowledge_inference", knowledge_inference_node,
                 metadata={"type": "agent", "llm_cfg": "config/knowledge_inference_llm_cfg.json"})

# ===== V5.3 新节点：多通道融合 / 品类模板 / 多语言OCR / 调用审计 =====
builder.add_node("multi_channel_fusion", multi_channel_fusion_node)
builder.add_node("category_template", category_template_node)
builder.add_node("multi_language_ocr", multi_language_ocr_node)
builder.add_node("call_audit", call_audit_node)

# 设置入口点（路由节点）
builder.set_entry_point("route_processing")

# 添加条件分支：路由处理模式
builder.add_conditional_edges(
    source="route_processing",
    path=route_processing_condition,
    path_map={
        "批量处理": "batch_process",
        "单图处理": "image_preprocess"
    }
)

# 批量处理 -> 结束
builder.add_edge("batch_process", END)

# ===== 单图处理流程 =====
# 1) 主OCR通道：图片预处理 -> OCR识别 -> 智能纠错 -> 结构化提取
builder.add_edge("image_preprocess", "ocr_recognize")
builder.add_edge("ocr_recognize", "correct_text")
builder.add_edge("correct_text", "model_extract")

# 2) VL并行通道：与model_extract并行，从image_preprocess出发
#    VL结果与OCR结果在 multi_channel_fusion 节点融合
builder.add_edge("image_preprocess", "vl_packaging_understanding")

# 3) 多通道融合：OCR + VL字段级加权融合
#    model_extract 和 vl_packaging_understanding 都汇入 multi_channel_fusion
builder.add_edge("model_extract", "multi_channel_fusion")
builder.add_edge("vl_packaging_understanding", "multi_channel_fusion")

# 4) 知识推理 + 品类模板：基于融合结果做品类适配和知识补全
builder.add_edge("multi_channel_fusion", "knowledge_inference")
builder.add_edge("knowledge_inference", "category_template")

# 5) 问答 + 结果输出
builder.add_edge("category_template", "qa_answer")
builder.add_edge("qa_answer", "result_output")

# 6) 飞书通知（call_audit 暂未接入主流程以避免冲突）
builder.add_edge("result_output", "feishu_notify")
builder.add_edge("feishu_notify", END)

# 编译图
main_graph = builder.compile()
