"""图像质量与智能路由节点 - 判断图像质量并选择最优处理路径"""
import os
import json
import math
import logging
from typing import Literal, Optional, Dict, Any
from pydantic import BaseModel, Field
from PIL import Image, ImageStat
from io import BytesIO
import requests
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context
from graphs.state import QualityRouterInput, QualityRouterOutput

logger = logging.getLogger(__name__)


def _fetch_image(url: str) -> Optional[Image.Image]:
    """从URL加载图片"""
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            return Image.open(BytesIO(resp.content))
    except Exception as e:
        logger.warning(f"图片加载失败: {url} - {e}")
    return None


def _estimate_image_quality(img: Image.Image) -> Dict[str, Any]:
    """评估图像质量分数（0-100）"""
    w, h = img.size
    total_pixels = w * h

    # 1. 分辨率评分 (40分)
    min_dim = min(w, h)
    if min_dim >= 2000:
        resol_score = 40
    elif min_dim >= 1200:
        resol_score = 30
    elif min_dim >= 800:
        resol_score = 20
    elif min_dim >= 500:
        resol_score = 10
    else:
        resol_score = 0

    # 2. 清晰度评分 (30分) - 基于拉普拉斯方差
    gray = img.convert("L")
    from PIL import ImageFilter
    laplacian = gray.filter(ImageFilter.Kernel((3, 3),
        [-1, -1, -1, -1, 8, -1, -1, -1, -1], scale=1))
    stat = ImageStat.Stat(laplacian)
    var_value = stat.stddev[0] if stat.stddev else 0
    if var_value > 50:
        clarity_score = 30
    elif var_value > 30:
        clarity_score = 20
    elif var_value > 15:
        clarity_score = 10
    else:
        clarity_score = 0

    # 3. 亮度/对比度评分 (15分)
    gray_stat = ImageStat.Stat(gray)
    mean_brightness = gray_stat.mean[0] if gray_stat.mean else 128
    # 太暗(<40)或太亮(>220)都扣分
    if 60 <= mean_brightness <= 200:
        brightness_score = 15
    elif 40 <= mean_brightness <= 220:
        brightness_score = 10
    else:
        brightness_score = 5

    # 4. 文本密度评估 (15分) - 基于边缘检测
    edge_img = gray.filter(ImageFilter.Kernel((3, 3),
        [-1, -1, -1, -1, 8, -1, -1, -1, -1], scale=1))
    edge_stat = ImageStat.Stat(edge_img)
    edge_energy = edge_stat.mean[0] if edge_stat.mean else 0
    if edge_energy > 5:
        text_score = 15
    elif edge_energy > 2:
        text_score = 10
    else:
        text_score = 5

    total = resol_score + clarity_score + brightness_score + text_score

    return {
        "total_score": total,
        "resolution_score": resol_score,
        "clarity_score": clarity_score,
        "brightness_score": brightness_score,
        "text_density_score": text_score,
        "width": w,
        "height": h,
        "laplacian_variance": round(var_value, 2),
        "mean_brightness": round(mean_brightness, 1)
    }


def _detect_possible_language(img: Image.Image) -> str:
    """基于图像属性猜测主要语言（供路由参考）"""
    return "zh"  # 默认中文，后续用实际检测模型


def image_quality_router_node(
    state: QualityRouterInput,
    config: RunnableConfig,
    runtime: Runtime[Context]
) -> QualityRouterOutput:
    """
    title: 图像质量评估与路由
    desc: 评估输入图像质量，智能选择最优处理路径：低质快速通道(VL-only)、高质精准通道(Full OCR+VL)、均衡通道(OCR-only)
    """
    ctx = runtime.context

    img = state.package_image

    # 加载图片
    pil_img = _fetch_image(img.url) if img.url.startswith("http") else None
    if pil_img is None:
        # 无法加载图片时走完整通道
        return QualityRouterOutput(
            selected_pipeline="full",
            quality_score=0,
            image_width=0,
            image_height=0,
            quality_detail={},
            auto_language="zh",
            pipeline_reason="无法加载图片，走完整OCR+VL通道"
        )

    # 评估质量
    quality = _estimate_image_quality(pil_img)
    score = quality["total_score"]

    w, h = quality["width"], quality["height"]
    aspect_ratio = w / h if h > 0 else 1

    # 路由决策逻辑
    if score >= 70 and aspect_ratio > 0.3 and aspect_ratio < 3.0:
        # 高质量 + 正常比例 → 完整管线（OCR+VL融合）
        pipeline = "full"
        reason = f"图像质量优秀({score}分)，走完整OCR+VL融合通道"
    elif aspect_ratio < 0.2 or aspect_ratio > 5.0:
        # 极端长宽比（可能是包装背面长图）→ OCR专线
        pipeline = "ocr_only"
        reason = f"极端长宽比({aspect_ratio:.2f})，走OCR专线"
    elif score < 25:
        # 低质量 → VL专线（跳过OCR）
        pipeline = "vl_only"
        reason = f"图像质量偏低({score}分)，走VL多模态专线(跳过OCR)"
    elif quality["laplacian_variance"] < 10:
        # 模糊 → 建议超分 + 完整管线
        pipeline = "enhance_full"
        reason = f"图像模糊(拉普拉斯方差={quality['laplacian_variance']:.1f})，先增强再走完整管线"
    else:
        # 中等质量 → 完整管线
        pipeline = "full"
        reason = f"图像质量中等({score}分)，走完整OCR+VL融合通道"

    logger.info(f"路由决策: {pipeline} | 质量分={score} | 尺寸={w}x{h} | 原因={reason}")

    return QualityRouterOutput(
        selected_pipeline=pipeline,
        quality_score=score,
        image_width=w,
        image_height=h,
        quality_detail=quality,
        auto_language=_detect_possible_language(pil_img),
        pipeline_reason=reason
    )