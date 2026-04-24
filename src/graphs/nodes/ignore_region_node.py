# -*- coding: utf-8 -*-
"""
忽略区域配置节点（V1.1新增）
支持排除水印、LOGO、页眉页脚等干扰区域
"""

import os
import traceback
from typing import Dict, Any, List, Optional
from datetime import datetime
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context

from graphs.state import (
    IgnoreRegionInput,
    IgnoreRegionOutput
)


class IgnoreRegionManager:
    """忽略区域管理器"""

    def __init__(self):
        self.regions = []

    def add_region(self, x1: int, y1: int, x2: int, y2: int):
        """添加忽略区域"""
        self.regions.append({
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2
        })

    def is_point_in_region(self, x: float, y: float, region: Dict[str, int]) -> bool:
        """检查点是否在区域内"""
        return (region["x1"] <= x <= region["x2"] and
                region["y1"] <= y <= region["y2"])

    def filter_ocr_result(self, ocr_regions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """过滤OCR结果"""
        if not self.regions:
            return ocr_regions

        filtered = []
        for item in ocr_regions:
            # 获取文本框坐标
            box = item.get("box") or item.get("bbox")

            if not box or len(box) < 4:
                filtered.append(item)
                continue

            # 计算文本框中心点
            center_x = (box[0] + box[2]) / 2
            center_y = (box[1] + box[3]) / 2

            # 检查中心点是否在忽略区域内
            is_ignored = False
            for region in self.regions:
                if self.is_point_in_region(center_x, center_y, region):
                    is_ignored = True
                    break

            if not is_ignored:
                filtered.append(item)

        return filtered

    def auto_detect_watermark(self, ocr_regions: List[Dict[str, Any]], threshold: float = 0.9) -> List[Dict[str, Any]]:
        """自动检测水印"""
        detected_watermarks = []

        # 水印特征：
        # 1. 文本较小
        # 2. 置信度较低
        # 3. 通常位于图片边缘
        # 4. 文本内容可能是水印关键词

        watermark_keywords = ["watermark", "水印", "sample", "样本", "demo", "演示", "preview", "预览"]

        for item in ocr_regions:
            # 检查文本内容
            text = item.get("text") or item.get("rec_text") or ""
            if any(keyword in text.lower() for keyword in watermark_keywords):
                box = item.get("box") or item.get("bbox")
                if box and len(box) >= 4:
                    detected_watermarks.append({
                        "x1": int(box[0]),
                        "y1": int(box[1]),
                        "x2": int(box[2]),
                        "y2": int(box[3]),
                        "text": text,
                        "reason": "keyword_match"
                    })
                continue

            # 检查置信度
            confidence = item.get("confidence") or item.get("score") or 1.0
            if confidence < threshold:
                box = item.get("box") or item.get("bbox")
                if box and len(box) >= 4:
                    detected_watermarks.append({
                        "x1": int(box[0]),
                        "y1": int(box[1]),
                        "x2": int(box[2]),
                        "y2": int(box[3]),
                        "text": text,
                        "reason": "low_confidence",
                        "confidence": confidence
                    })

        return detected_watermarks


def ignore_region_node(state: IgnoreRegionInput, config: RunnableConfig, runtime: Runtime[Context]) -> IgnoreRegionOutput:
    """
    title: 忽略区域配置
    desc: 支持排除水印、LOGO、页眉页脚等干扰区域，提升识别准确率
    integrations: -
    """
    ctx = runtime.context

    print(f"[忽略区域配置] 开始过滤OCR结果...")
    print(f"[忽略区域配置] 配置: 忽略区域数={len(state.ignore_regions)}, 自动检测水印={state.auto_detect_watermark}")

    try:
        start_time = datetime.now()

        manager = IgnoreRegionManager()

        # 添加用户指定的忽略区域
        for region in state.ignore_regions:
            manager.add_region(region["x1"], region["y1"], region["x2"], region["y2"])

        # 自动检测水印
        detected_watermarks = []
        if state.auto_detect_watermark:
            print(f"[忽略区域配置] 自动检测水印...")
            detected_watermarks = manager.auto_detect_watermark(state.ocr_regions, state.watermark_threshold)

            # 将检测到的水印添加到忽略区域
            for watermark in detected_watermarks:
                manager.add_region(
                    watermark["x1"],
                    watermark["y1"],
                    watermark["x2"],
                    watermark["y2"]
                )

            print(f"[忽略区域配置] 检测到 {len(detected_watermarks)} 个潜在水印区域")

        # 过滤OCR结果
        filtered_regions = manager.filter_ocr_result(state.ocr_regions)
        ignored_count = len(state.ocr_regions) - len(filtered_regions)

        # 提取过滤后的文本
        filtered_text = ""
        for region in filtered_regions:
            text = region.get("text") or region.get("rec_text") or ""
            if text:
                filtered_text += text + "\n"

        processing_time = (datetime.now() - start_time).total_seconds()

        print(f"[忽略区域配置] 过滤完成，耗时: {processing_time:.2f}秒")
        print(f"[忽略区域配置] 原始区域数: {len(state.ocr_regions)}, 过滤后: {len(filtered_regions)}, 忽略: {ignored_count}")

        return IgnoreRegionOutput(
            filtered_regions=filtered_regions,
            filtered_text=filtered_text.strip(),
            ignored_count=ignored_count,
            detected_watermarks=detected_watermarks,
            processing_time=processing_time
        )

    except Exception as e:
        print(f"[忽略区域配置] 处理失败: {e}")
        traceback.print_exc()
        raise
