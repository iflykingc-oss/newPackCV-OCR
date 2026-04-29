#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试OCR识别功能 - 使用有效图片URL
"""

import sys
import os
sys.path.insert(0, '/workspace/projects')
sys.path.insert(0, '/workspace/projects/src')

from graphs.state import OCRRecognizeInput
from utils.file.file import File
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context
from graphs.nodes.ocr_recognize_node import ocr_recognize_node

# 使用一个包含中文文字的公开图片URL
test_image_url = "https://images.unsplash.com/photo-1586281380349-632531db7ed4?w=800"  # 简单的产品图片

# 创建测试输入
test_input = OCRRecognizeInput(
    package_image=File(url=test_image_url),
    preprocessed_image=File(url=test_image_url),
    ocr_engine_type="builtin",
    ocr_api_config=None
)

# 创建配置和runtime
config = RunnableConfig(
    metadata={"llm_cfg": "config/model_extract_llm_cfg.json"}
)

# 创建一个简单的context
from coze_coding_utils.runtime_ctx.context import Context as Ctx
ctx = Ctx(
    run_id="test_run_123",
    space_id="7632237404564799542",
    project_id="7632238841302270002"
)

runtime = Runtime[Context](context=ctx)

print("开始测试OCR识别...")
print(f"测试图片URL: {test_image_url}\n")

result = ocr_recognize_node(test_input, config, runtime)

print(f"\n=== OCR识别结果 ===")
print(f"engine_used: {result.engine_used}")
print(f"raw_text: {result.raw_text}")
print(f"confidence: {result.confidence}")
print(f"processing_time: {result.processing_time}")
print(f"regions数量: {len(result.regions)}")

if result.regions:
    print(f"\n识别到的文字区域:")
    for i, region in enumerate(result.regions[:5]):  # 只显示前5个
        print(f"  {i+1}. '{region['text']}' (置信度: {region['confidence']:.2f})")
