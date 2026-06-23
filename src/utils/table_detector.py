"""表格检测与结构化提取

基于 OpenCV 的轻量级表格检测：
1. HoughLinesP 检测水平/垂直线 → 定位表格边界
2. 线交点 → 划分单元格
3. RapidOCR 逐单元格识别 → 结构化输出

集成到 ocr_postprocess.py 的营养成分表处理流程中。
"""

import os
import re
import logging
from typing import List, Tuple, Dict, Optional, Any

logger = logging.getLogger(__name__)


def detect_table_lines(image_input) -> Dict[str, Any]:
    """检测图像中的表格线，返回表格区域和行列分割信息

    Args:
        image_input: 文件路径(str) 或 numpy数组(np.ndarray)
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return {"success": False, "error": "opencv not installed"}

    if isinstance(image_input, str):
        img = cv2.imread(image_input)
    elif hasattr(image_input, 'shape'):
        img = image_input
    else:
        return {"success": False, "error": "invalid image input"}

    if img is None:
        return {"success": False, "error": "cannot read image"}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 自适应二值化
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5
    )

    # 检测水平线
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 8, 40), 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel, iterations=2)

    # 检测垂直线
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(h // 8, 40)))
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel, iterations=2)

    # 合并水平线和垂直线
    table_mask = cv2.add(h_lines, v_lines)

    # 膨胀使线条连接
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    table_mask = cv2.dilate(table_mask, dilate_kernel, iterations=2)

    # 查找轮廓 → 表格边界
    contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    tables = []
    min_area = (w * h) * 0.01  # 最小表格面积：图像的1%
    max_area = (w * h) * 0.95  # 最大表格面积：图像的95%

    for contour in contours:
        x, y, tw, th = cv2.boundingRect(contour)
        area = tw * th
        if min_area < area < max_area and tw > 50 and th > 50:
            # 提取表格区域的线
            roi_h = h_lines[y:y+th, x:x+tw]
            roi_v = v_lines[y:y+th, x:x+tw]

            # 用 HoughLinesP 精确检测线
            h_edges = cv2.Canny(roi_h, 50, 150)
            v_edges = cv2.Canny(roi_v, 50, 150)

            h_segs = cv2.HoughLinesP(h_edges, 1, 3.14/180, 100,
                                      minLineLength=tw//4, maxLineGap=10)
            v_segs = cv2.HoughLinesP(v_edges, 1, 3.14/180, 100,
                                      minLineLength=th//4, maxLineGap=10)

            # 提取行Y坐标和列X坐标
            y_coords = sorted(set([0, th] + [seg[0][1] for seg in h_segs] if h_segs is not None else [0, th]))
            x_coords = sorted(set([0, tw] + [seg[0][0] for seg in v_segs] if v_segs is not None else [0, tw]))

            # 合并相近的坐标（阈值：5像素）
            y_coords = _merge_close_coords(y_coords, 5)
            x_coords = _merge_close_coords(x_coords, 5)

            if len(y_coords) >= 2 and len(x_coords) >= 2:
                tables.append({
                    "bbox": {"x": x, "y": y, "w": tw, "h": th},
                    "rows": len(y_coords) - 1,
                    "cols": len(x_coords) - 1,
                    "y_coords": y_coords,
                    "x_coords": x_coords,
                })

    return {
        "success": True,
        "image_size": {"w": w, "h": h},
        "tables": tables,
        "table_count": len(tables),
    }


def _merge_close_coords(coords: List[int], threshold: int = 5) -> List[int]:
    """合并相近的坐标"""
    if not coords:
        return coords
    merged = [coords[0]]
    for c in coords[1:]:
        if c - merged[-1] > threshold:
            merged.append(c)
    return merged


def segment_table_cells(image_input, table: Dict[str, Any]) -> List[Dict[str, Any]]:
    """将表格区域分割成单元格"""
    try:
        import cv2
    except ImportError:
        return []

    if isinstance(image_input, str):
        img = cv2.imread(image_input)
    elif hasattr(image_input, 'shape'):
        img = image_input
    else:
        return []

    if img is None:
        return []

    bbox = table["bbox"]
    y_coords = table["y_coords"]
    x_coords = table["x_coords"]

    cells = []
    for row_idx in range(len(y_coords) - 1):
        for col_idx in range(len(x_coords) - 1):
            cy1 = bbox["y"] + y_coords[row_idx]
            cy2 = bbox["y"] + y_coords[row_idx + 1]
            cx1 = bbox["x"] + x_coords[col_idx]
            cx2 = bbox["x"] + x_coords[col_idx + 1]

            # 留2像素边距避免边框线干扰OCR
            pad = 2
            cell_img = img[max(0, cy1+pad):min(img.shape[0], cy2-pad),
                          max(0, cx1+pad):min(img.shape[1], cx2-pad)]

            if cell_img.size == 0 or cell_img.shape[0] < 5 or cell_img.shape[1] < 5:
                continue

            # 检查单元格是否包含有效内容（非纯白/纯黑）
            mean_val = cell_img.mean()
            if mean_val > 250 or mean_val < 5:
                continue

            cells.append({
                "row": row_idx,
                "col": col_idx,
                "bbox": {"x": cx1, "y": cy1, "w": cx2-cx1, "h": cy2-cy1},
                "image": cell_img,
            })

    return cells


def ocr_table_cells(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """对每个单元格执行 OCR"""
    try:
        from rapidocr_onnxruntime import RapidOCR
        engine = RapidOCR()
    except ImportError:
        return cells

    for cell in cells:
        img = cell.get("image")
        if img is None:
            cell["text"] = ""
            continue

        try:
            result, _ = engine(img)
            if result:
                texts = [line for _, line, conf in result]
                cell["text"] = " ".join(texts).strip()
                cell["confidence"] = sum(conf for _, _, conf in result) / len(result)
            else:
                cell["text"] = ""
                cell["confidence"] = 0.0
        except Exception as e:
            cell["text"] = ""
            cell["confidence"] = 0.0
            cell["error"] = str(e)

    return cells


def reconstruct_table(cells: List[Dict[str, Any]], rows: int, cols: int) -> List[List[str]]:
    """将单元格重建为二维表格"""
    grid = [["" for _ in range(cols)] for _ in range(rows)]
    for cell in cells:
        r, c = cell["row"], cell["col"]
        if r < rows and c < cols:
            grid[r][c] = cell.get("text", "")
    return grid


def format_table_as_text(grid: List[List[str]], delimiter: str = " | ") -> str:
    """将二维表格格式化为可读文本"""
    if not grid:
        return ""

    # 计算每列最大宽度
    col_widths = [0] * len(grid[0])
    for row in grid:
        for i, cell in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(cell))

    lines = []
    for row in grid:
        parts = []
        for i, cell in enumerate(row):
            width = col_widths[i] if i < len(col_widths) else len(cell)
            parts.append(cell.ljust(width))
        lines.append(delimiter.join(parts))

    return "\n".join(lines)


def detect_and_parse_table(image_input) -> Dict[str, Any]:
    """端到端：检测表格 → 分割单元格 → OCR → 结构化输出"""
    # 1. 检测表格
    detection = detect_table_lines(image_input)
    if not detection["success"] or detection["table_count"] == 0:
        return {"success": False, "error": "no table detected", "detection": detection}

    results = []
    for table in detection["tables"]:
        # 2. 分割单元格
        cells = segment_table_cells(image_input, table)
        if not cells:
            continue

        # 3. OCR 逐单元格
        cells = ocr_table_cells(cells)

        # 4. 重建表格
        grid = reconstruct_table(cells, table["rows"], table["cols"])
        text = format_table_as_text(grid)

        results.append({
            "bbox": table["bbox"],
            "rows": table["rows"],
            "cols": table["cols"],
            "grid": grid,
            "text": text,
            "cell_count": len(cells),
            "avg_confidence": sum(c.get("confidence", 0) for c in cells) / max(len(cells), 1),
        })

    return {
        "success": len(results) > 0,
        "tables": results,
        "table_count": len(results),
    }


def extract_nutrition_table(image_input) -> Optional[str]:
    """专门提取营养成分表（简化接口）"""
    result = detect_and_parse_table(image_input)
    if not result["success"]:
        return None

    # 在检测到的表格中查找包含营养成分关键词的表格
    nutrition_keywords = ['能量', '蛋白质', '脂肪', '碳水', '钠', '膳食',
                          'Energy', 'Protein', 'Fat', 'Carbohydrate', 'Sodium',
                          'Calories', 'Fiber', 'Sugar', 'Nutrition']

    for table in result["tables"]:
        text = table["text"]
        if any(kw in text for kw in nutrition_keywords):
            return text

    # 如果没有明确的营养成分表，返回最大的表格
    if result["tables"]:
        largest = max(result["tables"], key=lambda t: t["cell_count"])
        return largest["text"]

    return None
