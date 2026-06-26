# -*- coding: utf-8 -*-
"""节点模块初始化"""

from graphs.nodes.input_router_node import input_router_node
from graphs.nodes.scenario_detector_node import scenario_detector_node
from graphs.nodes.image_preprocess_node import image_preprocess_node
from graphs.nodes.ocr_recognize_node import ocr_recognize_node
from graphs.nodes.info_extract_node import info_extract_node
from graphs.nodes.qa_answer_node import qa_answer_node
from graphs.nodes.result_output_node import result_output_node
# VL多模态节点
from graphs.nodes.vl_scenario_detector_node import vl_scenario_detector_node
from graphs.nodes.vl_info_extract_node import vl_info_extract_node
# 批量处理节点
from graphs.nodes.batch_process_node import batch_process_node
# 多通道融合节点
from graphs.nodes.multi_channel_fusion_node import multi_channel_fusion_node

__all__ = [
    'input_router_node',
    'scenario_detector_node',
    'image_preprocess_node',
    'ocr_recognize_node',
    'info_extract_node',
    'qa_answer_node',
    'result_output_node',
    # VL多模态节点
    'vl_scenario_detector_node',
    'vl_info_extract_node',
    # 批量处理节点
    'batch_process_node',
    # 多通道融合节点
    'multi_channel_fusion_node'
]