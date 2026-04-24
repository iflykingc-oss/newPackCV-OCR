# -*- coding: utf-8 -*-
"""
智能排版解析节点（V1.1新增）
支持多栏布局识别、自然段换行、保留缩进等功能
"""

import os
import traceback
from typing import Dict, Any, List, Optional
from datetime import datetime
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context

from graphs.state import (
    LayoutParseInput,
    LayoutParseOutput
)


class LayoutParser:
    """排版解析器"""

    def detect_layout(self, ocr_regions: List[Dict[str, Any]]) -> str:
        """检测布局类型（单栏/多栏）"""
        import numpy as np

        if not ocr_regions:
            return "single_column"

        # 提取文本框位置
        boxes = []
        for region in ocr_regions:
            if "box" in region:
                boxes.append(region["box"])
            elif "bbox" in region:
                boxes.append(region["bbox"])

        if not boxes:
            return "single_column"

        # 分析X坐标分布
        x_positions = []
        for box in boxes:
            if len(box) >= 4:
                # 取文本框的中心X坐标
                center_x = (box[0] + box[2]) / 2
                x_positions.append(center_x)

        if not x_positions:
            return "single_column"

        # 计算X坐标的标准差
        x_std = np.std(x_positions)

        # 如果标准差较大，说明是多栏布局
        # 阈值可以根据实际情况调整
        threshold = 100  # 像素
        if x_std > threshold:
            return "multi_column"
        else:
            return "single_column"

    def parse_single_column(self, ocr_regions: List[Dict[str, Any]], enable_paragraph_break: bool) -> str:
        """解析单栏布局"""
        lines = []

        # 按Y坐标排序
        sorted_regions = self._sort_by_y_position(ocr_regions)

        for i, region in enumerate(sorted_regions):
            text = self._extract_text(region)

            if not text:
                continue

            # 自然段换行逻辑
            if enable_paragraph_break:
                # 检查是否是段落结束
                if i > 0:
                    prev_region = sorted_regions[i - 1]
                    prev_y = self._get_y_position(prev_region)
                    curr_y = self._get_y_position(region)

                    # 如果Y坐标间距较大，认为是新段落
                    gap = curr_y - prev_y
                    line_height = self._estimate_line_height(region)

                    if gap > line_height * 1.5:
                        lines.append("")  # 添加空行表示段落

            lines.append(text)

        return "\n".join(lines)

    def parse_multi_column(self, ocr_regions: List[Dict[str, Any]], enable_paragraph_break: bool) -> str:
        """解析多栏布局"""
        import numpy as np

        # 1. 识别栏数
        columns = self._identify_columns(ocr_regions)

        # 2. 按栏分组
        column_groups = self._group_by_columns(ocr_regions, columns)

        # 3. 分别解析每一栏
        column_texts = []
        for group in column_groups:
            sorted_group = self._sort_by_y_position(group)
            lines = []
            for region in sorted_group:
                text = self._extract_text(region)
                if text:
                    lines.append(text)
            column_texts.append("\n".join(lines))

        # 4. 合并多栏文本（可以按栏顺序或按行顺序）
        # 这里简单按栏顺序合并
        result = "\n\n".join(column_texts)

        return result

    def parse_preserve_indent(self, ocr_regions: List[Dict[str, Any]]) -> str:
        """解析并保留缩进"""
        lines = []
        sorted_regions = self._sort_by_y_position(ocr_regions)

        for region in sorted_regions:
            text = self._extract_text(region)
            box = region.get("box") or region.get("bbox")

            if text and box and len(box) >= 4:
                # 计算缩进（基于X坐标）
                indent = int(box[0] / 10)  # 假设每个缩进是10像素
                indent_str = " " * indent
                lines.append(indent_str + text)
            elif text:
                lines.append(text)

        return "\n".join(lines)

    def _sort_by_y_position(self, ocr_regions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """按Y坐标排序"""
        def get_y(region):
            box = region.get("box") or region.get("bbox")
            if box and len(box) >= 4:
                return box[1]
            return 0

        return sorted(ocr_regions, key=get_y)

    def _extract_text(self, region: Dict[str, Any]) -> str:
        """提取文本"""
        if "text" in region:
            return region["text"]
        elif "rec_text" in region:
            return region["rec_text"]
        return ""

    def _get_y_position(self, region: Dict[str, Any]) -> float:
        """获取Y坐标"""
        box = region.get("box") or region.get("bbox")
        if box and len(box) >= 4:
            return box[1]
        return 0

    def _estimate_line_height(self, region: Dict[str, Any]) -> float:
        """估算行高"""
        box = region.get("box") or region.get("bbox")
        if box and len(box) >= 4:
            return box[3] - box[1]
        return 20  # 默认行高

    def _identify_columns(self, ocr_regions: List[Dict[str, Any]]) -> int:
        """识别栏数"""
        import numpy as np
        from sklearn.cluster import KMeans

        # 提取X坐标
        x_positions = []
        for region in ocr_regions:
            box = region.get("box") or region.get("bbox")
            if box and len(box) >= 4:
                center_x = (box[0] + box[2]) / 2
                x_positions.append([center_x])

        if not x_positions:
            return 1

        # 使用K-means聚类识别栏数
        try:
            # 尝试2-5栏
            for n_clusters in range(2, 6):
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                kmeans.fit(x_positions)

                # 计算聚类内方差
                inertia = kmeans.inertia_

                # 如果聚类效果明显，返回该栏数
                if inertia < len(x_positions) * 100:  # 阈值
                    return n_clusters
        except:
            pass

        return 1

    def _group_by_columns(self, ocr_regions: List[Dict[str, Any]], column_count: int) -> List[List[Dict[str, Any]]]:
        """按栏分组"""
        import numpy as np
        from sklearn.cluster import KMeans

        # 提取X坐标
        x_positions = []
        for region in ocr_regions:
            box = region.get("box") or region.get("bbox")
            if box and len(box) >= 4:
                center_x = (box[0] + box[2]) / 2
                x_positions.append([center_x])

        if not x_positions or column_count == 1:
            return [ocr_regions]

        # 使用K-means聚类
        try:
            kmeans = KMeans(n_clusters=column_count, random_state=42, n_init=10)
            labels = kmeans.fit_predict(x_positions)

            # 按标签分组
            groups = [[] for _ in range(column_count)]
            for i, region in enumerate(ocr_regions):
                if i < len(labels):
                    groups[labels[i]].append(region)

            # 按栏的X坐标排序
            centers = kmeans.cluster_centers_.flatten()
            sorted_indices = np.argsort(centers)

            sorted_groups = [groups[i] for i in sorted_indices]

            return sorted_groups
        except:
            return [ocr_regions]


def layout_parse_node(state: LayoutParseInput, config: RunnableConfig, runtime: Runtime[Context]) -> LayoutParseOutput:
    """
    title: 智能排版解析
    desc: 支持多栏布局识别、自然段换行、保留缩进等功能，输出格式化文本
    integrations: scikit-learn
    """
    ctx = runtime.context

    print(f"[智能排版解析] 开始解析排版...")
    print(f"[智能排版解析] 配置: 解析模式={state.parse_mode}, 自然段换行={state.enable_paragraph_break}, 竖排文本={state.enable_vertical_text}")

    try:
        import numpy as np

        start_time = datetime.now()

        parser = LayoutParser()

        # 根据解析模式选择策略
        if state.parse_mode == "auto":
            # 自动检测布局
            layout_type = parser.detect_layout(state.ocr_regions)
            print(f"[智能排版解析] 检测到布局类型: {layout_type}")

            if layout_type == "multi_column":
                parsed_text = parser.parse_multi_column(state.ocr_regions, state.enable_paragraph_break)
                column_count = parser._identify_columns(state.ocr_regions)
            else:
                parsed_text = parser.parse_single_column(state.ocr_regions, state.enable_paragraph_break)
                column_count = 1

        elif state.parse_mode == "multi_column":
            parsed_text = parser.parse_multi_column(state.ocr_regions, state.enable_paragraph_break)
            column_count = parser._identify_columns(state.ocr_regions)
            layout_type = "multi_column"

        elif state.parse_mode == "single_column":
            parsed_text = parser.parse_single_column(state.ocr_regions, state.enable_paragraph_break)
            column_count = 1
            layout_type = "single_column"

        elif state.parse_mode == "preserve_indent":
            parsed_text = parser.parse_preserve_indent(state.ocr_regions)
            column_count = 1
            layout_type = "single_column_indent"

        else:
            # 默认单栏
            parsed_text = parser.parse_single_column(state.ocr_regions, state.enable_paragraph_break)
            column_count = 1
            layout_type = "single_column"

        # 统计段落数
        paragraph_count = len([line for line in parsed_text.split('\n') if line.strip()])

        processing_time = (datetime.now() - start_time).total_seconds()

        print(f"[智能排版解析] 解析完成，耗时: {processing_time:.2f}秒")
        print(f"[智能排版解析] 布局: {layout_type}, 栏数: {column_count}, 段落数: {paragraph_count}")

        return LayoutParseOutput(
            parsed_text=parsed_text,
            layout_type=layout_type,
            paragraph_count=paragraph_count,
            column_count=column_count,
            processing_time=processing_time
        )

    except Exception as e:
        print(f"[智能排版解析] 处理失败: {e}")
        traceback.print_exc()
        raise
