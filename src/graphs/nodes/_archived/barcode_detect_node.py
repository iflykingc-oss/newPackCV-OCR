# -*- coding: utf-8 -*-
"""
条码/二维码检测节点
使用pyzbar库检测图片中的条码和二维码
V5.9 新增
"""

import os
import tempfile
import logging
from typing import List, Dict, Any
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context
from graphs.state import BarcodeDetectInput, BarcodeDetectOutput
from utils.file.file import FileOps

logger = logging.getLogger(__name__)

try:
    from pyzbar.pyzbar import decode as pyzbar_decode
    from PIL import Image as PILImage
    PYZBAR_AVAILABLE = True
except ImportError:
    PYZBAR_AVAILABLE = False
    logger.warning("pyzbar未安装，条码检测将返回空结果")


def _decode_barcodes(image_path: str, barcode_types: List[str]) -> List[Dict[str, Any]]:
    """调用pyzbar解码条码

    Args:
        image_path: 图片本地路径
        barcode_types: 启用的条码类型列表

    Returns:
        条码检测结果列表
    """
    if not PYZBAR_AVAILABLE:
        return []

    results = []
    try:
        pil_img = PILImage.open(image_path)
        decoded_objects = pyzbar_decode(pil_img)

        for obj in decoded_objects:
            barcode_type = obj.type
            # 类型过滤
            if barcode_types and barcode_type.lower() not in {t.lower() for t in barcode_types}:
                continue

            data = obj.data.decode("utf-8", errors="replace")
            rect = obj.rect
            polygon = obj.polygon

            # 计算置信度 (pyzbar不提供置信度，基于分辨率估算)
            confidence = 0.85 if rect.width > 20 and rect.height > 20 else 0.6

            results.append({
                "type": barcode_type,
                "data": data,
                "rect": {
                    "x": rect.left,
                    "y": rect.top,
                    "width": rect.width,
                    "height": rect.height,
                },
                "polygon": [(p.x, p.y) for p in polygon] if polygon else [],
                "confidence": confidence,
            })
    except Exception as e:
        logger.error("条码解码失败: %s", e)

    return results


def barcode_detect_node(
    state: BarcodeDetectInput,
    config: RunnableConfig,
    runtime: Runtime[Context],
) -> BarcodeDetectOutput:
    """
    title: 条码/二维码检测
    desc: 检测图片中的条码(EAN13/Code128/Code39等)和二维码(QR码)，返回条码类型、数据内容和位置
    integrations: 
    """
    ctx = runtime.context

    if not PYZBAR_AVAILABLE:
        logger.warning("pyzbar未安装，跳过条码检测")
        return BarcodeDetectOutput(
            barcodes=[],
            total_found=0,
            has_barcode=False,
        )

    # 下载图片到临时目录
    tmp_dir = tempfile.mkdtemp(prefix="barcode_")
    try:
        local_path = FileOps.save_to_local(state.package_image.url)
        if local_path is None:
            raise ValueError("下载图片失败")

        # 解码条码
        barcode_types = state.barcode_types or [
            "qr", "code128", "ean13", "ean8", "upca", "code39", "itf", "pdf417"
        ]
        barcodes = _decode_barcodes(local_path, barcode_types)

        logger.info("条码检测完成: 共%d个条码", len(barcodes))

        return BarcodeDetectOutput(
            barcodes=barcodes,
            total_found=len(barcodes),
            has_barcode=len(barcodes) > 0,
        )
    except Exception as e:
        logger.error("条码检测异常: %s", e)
        return BarcodeDetectOutput(
            barcodes=[],
            total_found=0,
            has_barcode=False,
        )
    finally:
        # 清理临时文件
        try:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass