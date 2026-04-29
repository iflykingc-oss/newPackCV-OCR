#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试OCR识别功能
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

# 创建测试输入
test_input = OCRRecognizeInput(
    package_image=File(
        url="https://code.coze.cn/api/sandbox/coze_coding/file/proxy?expire_time=-1&file_path=assets%2F11_20260429101949948.jpg"
    ),
    preprocessed_image=File(
        url="https://code.coze.cn/api/sandbox/coze_coding/file/proxy?expire_time=-1&file_path=assets%2F11_20260429101949948.jpg"
    ),
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
result = ocr_recognize_node(test_input, config, runtime)

print(f"\n=== OCR识别结果 ===")
print(f"engine_used: {result.engine_used}")
print(f"raw_text: {result.raw_text}")
print(f"confidence: {result.confidence}")
print(f"processing_time: {result.processing_time}")
print(f"regions数量: {len(result.regions)}")
