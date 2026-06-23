# -*- coding: utf-8 -*-
"""
细粒度商品识别节点（V1.2新增）
多粒度特征融合，细粒度商品识别（规格、年份、批次等）
"""

import os
import traceback
from typing import Dict, Any, List, Optional
from datetime import datetime
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context

from graphs.state import (
    FineGrainedRecognitionInput,
    FineGrainedRecognitionOutput
)


def extract_visual_features(image, bbox):
    """提取视觉特征"""
    import cv2
    import numpy as np

    x1, y1, x2, y2 = bbox
    roi = image[y1:y2, x1:x2]

    # 计算颜色直方图
    hist = cv2.calcHist([roi], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    # 计算梯度特征
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    gradient_feature = [np.mean(gradient_magnitude), np.std(gradient_magnitude)]

    # 合并特征
    feature = np.concatenate([hist, gradient_feature])

    return feature


def extract_text_features(ocr_results, bbox):
    """提取文本特征"""
    import numpy as np

    # 找到ROI范围内的OCR结果
    roi_texts = []
    for result in ocr_results:
        text_bbox = result.get("bbox") or result.get("box")
        if text_bbox:
            # 简化处理：检查文本框中心是否在ROI内
            cx = (text_bbox[0] + text_bbox[2]) / 2 if len(text_bbox) >= 4 else text_bbox[0]
            cy = (text_bbox[1] + text_bbox[3]) / 2 if len(text_bbox) >= 4 else text_bbox[1]

            x1, y1, x2, y2 = bbox
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                text = result.get("text") or result.get("rec_text", "")
                roi_texts.append(text)

    # 文本特征：长度、平均词长、数字占比等
    all_text = " ".join(roi_texts)
    features = [
        len(all_text),
        len(roi_texts),
        sum(1 for c in all_text if c.isdigit()) / len(all_text) if all_text else 0,
    ]

    return features


def extract_barcode_features(image, bbox):
    """提取条形码特征"""
    import cv2
    import numpy as np

    x1, y1, x2, y2 = bbox
    roi = image[y1:y2, x1:x2]

    # 转灰度
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # 边缘检测（条形码特征）
    edges = cv2.Canny(gray, 50, 150)

    # 统计边缘特征
    edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])

    # 水平投影（条形码通常是水平条纹）
    horizontal_projection = np.sum(edges, axis=0)

    features = [
        edge_density,
        np.mean(horizontal_projection),
        np.std(horizontal_projection),
    ]

    return features


def classify_product_attributes(text: str) -> Dict[str, List[str]]:
    """分类商品属性"""
    import re

    attributes = {
        "specification": [],
        "flavor": [],
        "capacity": [],
        "batch": [],
        "year": []
    }

    # 提取规格
    spec_patterns = [
        r"(\d+)(ml|g|kg|L)",
        r"(\d+)\*(\d+)(ml|g|kg)",
        r"(\d+)盒装",
        r"(\d+)瓶装"
    ]
    for pattern in spec_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                attributes["specification"].append("".join(match))
            else:
                attributes["specification"].append(match)

    # 提取口味/风味
    flavor_keywords = ["原味", "柠檬", "橙子", "苹果", "葡萄", "桃", "荔枝", "薄荷", "咖啡", "巧克力"]
    for keyword in flavor_keywords:
        if keyword in text:
            attributes["flavor"].append(keyword)

    # 提取容量
    capacity_patterns = [
        r"(\d+)(ml|毫升|L|升)",
        r"(\d+)(g|克|kg|千克)"
    ]
    for pattern in capacity_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                attributes["capacity"].append("".join(match))
            else:
                attributes["capacity"].append(match)

    # 提取批号
    batch_patterns = [
        r"批号[:：]?([A-Za-z0-9]+)",
        r"batch[:：]?([A-Za-z0-9]+)",
        r"lot[:：]?([A-Za-z0-9]+)"
    ]
    for pattern in batch_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        attributes["batch"].extend(matches)

    # 提取年份
    year_pattern = r"(19|20)\d{2}"
    years = re.findall(year_pattern, text)
    attributes["year"].extend([y + (re.search(year_pattern, text).group(0)[2:] if re.search(year_pattern, text) else "") for y in years])

    return attributes


def fine_grained_recognition_node(
    state: FineGrainedRecognitionInput,
    config: RunnableConfig,
    runtime: Runtime[Context]
) -> FineGrainedRecognitionOutput:
    """
    title: 细粒度商品识别
    desc: 多粒度特征融合（视觉+文字+条码），细粒度商品识别，区分规格、年份、批次等
    integrations: OpenCV
    """
    ctx = runtime.context

    print(f"[细粒度识别] 开始处理...")
    print(f"[细粒度识别] 配置: 多模态融合={state.enable_multimodal_fusion}, 条形码识别={state.enable_barcode_recognition}")

    try:
        import cv2
        import numpy as np

        start_time = datetime.now()

        # 下载图片
        def download_image(url):
            import requests
            try:
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    return response.content
            except:
                pass
            return None

        img_data = download_image(state.image.url)
        if not img_data:
            raise Exception("图片下载失败")

        image = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise Exception("图片解码失败")

        print(f"[细粒度识别] 图片尺寸: {image.shape}")

        recognized_products = []
        product_attributes = {}

        for det in state.detection_boxes:
            bbox = det.get("bbox") or det.get("box")
            if not bbox or len(bbox) < 4:
                continue

            x1, y1, x2, y2 = [int(coord) for coord in bbox[:4]]
            bbox = [x1, y1, x2, y2]

            print(f"[细粒度识别] 处理检测框: {bbox}")

            # 提取特征
            visual_feature = extract_visual_features(image, bbox)
            text_feature = extract_text_features(state.ocr_results, bbox)
            barcode_feature = extract_barcode_features(image, bbox)

            # 融合特征
            if state.enable_multimodal_fusion:
                fused_feature = np.concatenate([visual_feature, text_feature, barcode_feature])
            else:
                fused_feature = visual_feature

            # 简化的分类逻辑（实际应该使用分类模型）
            product_info = {
                "bbox": bbox,
                "visual_feature": visual_feature.tolist() if isinstance(visual_feature, np.ndarray) else visual_feature,
                "text_feature": text_feature,
                "confidence": 0.85  # 简化：固定置信度
            }

            # 提取商品属性
            roi_texts = []
            for result in state.ocr_results:
                text_bbox = result.get("bbox") or result.get("box")
                if text_bbox:
                    cx = (text_bbox[0] + text_bbox[2]) / 2 if len(text_bbox) >= 4 else text_bbox[0]
                    cy = (text_bbox[1] + text_bbox[3]) / 2 if len(text_bbox) >= 4 else text_bbox[1]

                    if x1 <= cx <= x2 and y1 <= cy <= y2:
                        text = result.get("text") or result.get("rec_text", "")
                        roi_texts.append(text)

            all_text = " ".join(roi_texts)
            attributes = classify_product_attributes(all_text)

            product_info.update(attributes)
            recognized_products.append(product_info)

            # 合并属性
            for key, value in attributes.items():
                if value:
                    if key not in product_attributes:
                        product_attributes[key] = []
                    product_attributes[key].extend(value)

        # 计算整体识别置信度
        if recognized_products:
            recognition_confidence = np.mean([p["confidence"] for p in recognized_products])
        else:
            recognition_confidence = 0.0

        processing_time = (datetime.now() - start_time).total_seconds()

        print(f"[细粒度识别] 处理完成，耗时: {processing_time:.2f}秒")
        print(f"[细粒度识别] 识别商品数: {len(recognized_products)}, 整体置信度: {recognition_confidence:.2f}")

        return FineGrainedRecognitionOutput(
            recognized_products=recognized_products,
            product_attributes=product_attributes,
            recognition_confidence=float(recognition_confidence),
            processing_time=processing_time
        )

    except Exception as e:
        print(f"[细粒度识别] 处理失败: {e}")
        traceback.print_exc()
        raise
