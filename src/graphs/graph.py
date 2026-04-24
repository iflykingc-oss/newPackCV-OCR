# -*- coding: utf-8 -*-
"""
多平台OCR包装识别系统 - 主图编排
构建完整的DAG工作流，支持图片预处理、OCR识别、模型调用、结果输出、批量处理
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


def route_processing_mode(state: GlobalState) -> str:
    """
    title: 路由处理模式
    desc: 根据输入决定是单图处理还是批量处理：如果images字段存在且有多个图片，则进入批量处理
    """
    # 检查是否有images字段（用于批量处理）
    images = getattr(state, 'images', None)
    
    if images and isinstance(images, list) and len(images) > 1:
        return "批量处理"
    else:
        return "单图处理"


def route_model_processing(state: GlobalState) -> str:
    """
    title: 路由模型处理
    desc: 根据model_type字段决定后续处理流程：extract(结构化提取)、correct(智能纠错)、qa(语义问答)
    """
    # 从状态中读取model_type
    model_type = getattr(state, 'model_type', None)
    
    # 默认使用extract
    if not model_type:
        model_type = "extract"
    
    if model_type == "extract":
        return "结构化提取"
    elif model_type == "correct":
        return "智能纠错"
    elif model_type == "qa":
        return "语义问答"
    else:
        return "结构化提取"  # 默认


def route_processing_node(state: RouteProcessingInput, config: RunnableConfig, runtime: Runtime[Context]) -> RouteProcessingOutput:
    """
    title: 路由处理模式
    desc: 根据输入决定是单图处理还是批量处理
    """
    ctx = runtime.context
    
    # 检查是否有images字段（用于批量处理）
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

# 批量处理 -> 结束（批量处理节点自己完成所有工作）
builder.add_edge("batch_process", END)

# 单图处理流程：图片预处理 -> OCR识别
builder.add_edge("image_preprocess", "ocr_recognize")

# OCR识别 -> 条件分支（根据model_type选择）
builder.add_conditional_edges(
    source="ocr_recognize",
    path=route_model_processing,
    path_map={
        "结构化提取": "model_extract",
        "智能纠错": "correct_text",
        "语义问答": "qa_answer"
    }
)

# 所有模型处理分支 -> 结果输出
builder.add_edge("model_extract", "result_output")
builder.add_edge("correct_text", "result_output")
builder.add_edge("qa_answer", "result_output")

# 结果输出 -> 结束
builder.add_edge("result_output", END)

# 编译图
main_graph = builder.compile()
