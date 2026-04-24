# -*- coding: utf-8 -*-
"""
PackCV-OCR 融合算法工作流编排
构建货架/多包装场景的完整CV+OCR处理流程
"""

from typing import Optional, Dict, Any, List
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context

from graphs.state import (
    GlobalState,
    PackCVGraphInput,
    PackCVGraphOutput,
    ParallelProcessingOutput,
    AlertEngineOutput
)

# 导入PackCV-OCR节点
from graphs.nodes.cv_detection_node import cv_detection_node
from graphs.nodes.roi_segmentation_node import roi_segmentation_node
from graphs.nodes.parallel_processing_node import parallel_processing_node
from graphs.nodes.alert_engine_node import alert_engine_node
from graphs.nodes.report_generation_node import report_generation_node


def route_expiry_analysis(state: GlobalState) -> str:
    """
    title: 路由效期分析
    desc: 根据enable_expiry_detection决定是否进行效期分析
    """
    enable_expiry = getattr(state, 'enable_expiry_detection', True)
    return "启用效期分析" if enable_expiry else "跳过效期分析"


def route_inventory_analysis(state: GlobalState) -> str:
    """
    title: 路由库存分析
    desc: 根据enable_inventory_analysis决定是否进行库存分析
    """
    enable_inventory = getattr(state, 'enable_inventory_analysis', True)
    return "启用库存分析" if enable_inventory else "跳过库存分析"


def route_alerts(state: GlobalState) -> str:
    """
    title: 路由告警
    desc: 根据enable_alerts决定是否生成告警
    """
    enable_alerts = getattr(state, 'enable_alerts', True)
    return "启用告警" if enable_alerts else "跳过告警"


# 创建状态图
builder = StateGraph(GlobalState, input_schema=PackCVGraphInput, output_schema=PackCVGraphOutput)

# 添加节点
builder.add_node("cv_detection", cv_detection_node)
builder.add_node("roi_segmentation", roi_segmentation_node)
builder.add_node("parallel_processing", parallel_processing_node)
builder.add_node("alert_engine", alert_engine_node)
builder.add_node("report_generation", report_generation_node)

# 设置入口点
builder.set_entry_point("cv_detection")

# 添加边
# CV检测 -> ROI裁切
builder.add_edge("cv_detection", "roi_segmentation")

# ROI裁切 -> 并行处理
builder.add_edge("roi_segmentation", "parallel_processing")

# 并行处理 -> 条件分支（是否启用告警）
builder.add_conditional_edges(
    source="parallel_processing",
    path=route_alerts,
    path_map={
        "启用告警": "alert_engine",
        "跳过告警": "report_generation"
    }
)

# 告警引擎 -> 报表生成
builder.add_edge("alert_engine", "report_generation")

# 报表生成 -> 结束
builder.add_edge("report_generation", END)

# 编译图
packcv_ocr_graph = builder.compile()
