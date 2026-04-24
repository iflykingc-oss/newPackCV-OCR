# -*- coding: utf-8 -*-
"""
多平台OCR包装识别系统 - 主图编排
构建完整的DAG工作流，支持图片预处理、OCR识别、模型调用、结果输出
"""

from langgraph.graph import StateGraph, END
from graphs.state import (
    GlobalState,
    GraphInput,
    GraphOutput,
    ResultOutputInput
)

# 导入所有节点
from graphs.nodes.image_preprocess_node import image_preprocess_node
from graphs.nodes.ocr_recognize_node import ocr_recognize_node
from graphs.nodes.model_extract_node import model_extract_node
from graphs.nodes.correct_text_node import correct_text_node
from graphs.nodes.qa_answer_node import qa_answer_node
from graphs.nodes.result_output_node import result_output_node


def route_model_processing(state: GlobalState) -> str:
    """
    title: 路由模型处理
    desc: 根据model_type字段决定后续处理流程：extract(结构化提取)、correct(智能纠错)、qa(语义问答)
    """
    # 从输入中获取model_type
    # 注意：这里需要从GraphInput中获取，因为这是工作流的初始输入
    # 但在运行时，state中已经包含了所有字段
    
    # 从状态中读取model_type（如果有）
    model_type = getattr(state, 'model_type', None)
    
    # 如果状态中没有，尝试从其他来源获取
    if not model_type:
        # 尝试从platform_config或其他地方读取
        pass
    
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


# 创建状态图
builder = StateGraph(GlobalState, input_schema=GraphInput, output_schema=GraphOutput)

# 添加节点
builder.add_node("image_preprocess", image_preprocess_node)
builder.add_node("ocr_recognize", ocr_recognize_node)
builder.add_node("model_extract", model_extract_node, 
                 metadata={"type": "agent", "llm_cfg": "config/model_extract_llm_cfg.json"})
builder.add_node("correct_text", correct_text_node,
                 metadata={"type": "agent", "llm_cfg": "config/correct_text_llm_cfg.json"})
builder.add_node("qa_answer", qa_answer_node,
                 metadata={"type": "agent", "llm_cfg": "config/qa_answer_llm_cfg.json"})
builder.add_node("result_output", result_output_node)

# 设置入口点
builder.set_entry_point("image_preprocess")

# 添加边
# 图片预处理 -> OCR识别
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
