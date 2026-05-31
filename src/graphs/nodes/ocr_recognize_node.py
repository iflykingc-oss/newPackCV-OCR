# -*- coding: utf-8 -*-
"""
OCR文字识别节点 - 稳定版
基于Tesseract OCR，支持中英文识别
"""
import os
import json
import time
import logging
import requests
from typing import List, Dict, Any, Optional
from io import BytesIO

import cv2
import numpy as np
from PIL import Image
import pytesseract

from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context
from graphs.state import OCRRecognizeInput, OCRRecognizeOutput
from utils.file.file import File
from coze_coding_dev_sdk.s3 import S3SyncStorage

logger = logging.getLogger(__name__)


def _get_s3_storage() -> S3SyncStorage:
    """获取S3对象存储客户端"""
    return S3SyncStorage(
        endpoint_url=os.getenv("COZE_BUCKET_ENDPOINT_URL"),
        access_key="",
        secret_key="",
        bucket_name=os.getenv("COZE_BUCKET_NAME"),
        region="cn-beijing",
    )


def download_image(url: str) -> Optional[np.ndarray]:
    """下载图片并转为OpenCV格式"""
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        pil_img = Image.open(BytesIO(resp.content))
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        logger.warning(f"图片下载失败: {e}")
        return None


def ocr_with_tesseract(img: np.ndarray, psm: int = 6, lang: str = "chi_sim+eng") -> str:
    """
    使用Tesseract进行OCR识别
    """
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
        pil_img = Image.fromarray(gray)
        config = f'--psm {psm} -l {lang} --oem 3'
        return pytesseract.image_to_string(pil_img, config=config).strip()
    except Exception as e:
        logger.warning(f"Tesseract OCR失败 (psm={psm}, lang={lang}): {e}")
        return ""


def preprocess_for_ocr(img: np.ndarray) -> np.ndarray:
    """
    OCR前预处理：灰度化 + 适度增强
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 轻度CLAHE增强
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return enhanced


def ocr_recognize_node(
    state: OCRRecognizeInput,
    config: RunnableConfig,
    runtime: Runtime[Context]
) -> OCRRecognizeOutput:
    """
    title: OCR文字识别
    desc: 使用Tesseract OCR进行文字识别，支持中英文。当预处理图片识别失败时自动回退到原始图片。
    integrations: Tesseract OCR
    """
    ctx = runtime.context
    start_time = time.time()

    # 选择图片（优先预处理后的图片）
    image_url = ""
    if state.preprocessed_image and state.preprocessed_image.url:
        image_url = state.preprocessed_image.url
    elif state.package_image and state.package_image.url:
        image_url = state.package_image.url
    elif state.image and state.image.url:
        image_url = state.image.url

    if not image_url:
        logger.error("无可用图片")
        return OCRRecognizeOutput(
            raw_text="[ERROR] 无可用图片",
            ocr_confidence=0.0,
            engine_used="none",
            processing_time=time.time() - start_time
        )

    # 下载图片
    img = download_image(image_url)
    if img is None:
        return OCRRecognizeOutput(
            raw_text="[ERROR] 图片下载失败",
            ocr_confidence=0.0,
            engine_used="none",
            processing_time=time.time() - start_time
        )

    # OCR识别
    final_text = ""
    final_conf = 0.0
    engine = "tesseract"

    # 1. 尝试预处理图片 OCR
    processed = preprocess_for_ocr(img)
    processed_bgr = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

    # PSM 6: 统一文本块
    final_text = ocr_with_tesseract(processed_bgr, psm=6, lang="chi_sim+eng")
    if not final_text:
        # PSM 3: 完全自动分页
        final_text = ocr_with_tesseract(processed_bgr, psm=3, lang="chi_sim+eng")
    if not final_text:
        # 仅用英文
        final_text = ocr_with_tesseract(processed_bgr, psm=6, lang="eng")

    # 2. 如果预处理图片OCR为空，尝试原始图片
    if not final_text:
        logger.info("预处理后OCR为空，尝试原始图片")
        original_url = ""
        if state.package_image and state.package_image.url:
            original_url = state.package_image.url
        elif state.image and state.image.url:
            original_url = state.image.url

        if original_url and original_url != image_url:
            original_img = download_image(original_url)
            if original_img is not None:
                final_text = ocr_with_tesseract(original_img, psm=6, lang="chi_sim+eng")
                if not final_text:
                    final_text = ocr_with_tesseract(original_img, psm=3, lang="chi_sim+eng")
                if final_text:
                    engine = "tesseract_original"

    # 3. 如果仍然为空，标记为失败但返回空字符串
    if not final_text:
        logger.warning("OCR识别结果为空")
        final_text = ""
        final_conf = 0.0
    else:
        final_conf = 0.75

    elapsed_time = time.time() - start_time
    logger.info(f"OCR完成: 耗时={elapsed_time:.2f}s, 文本长度={len(final_text)}")

    # 上传结果到S3
    try:
        storage = _get_s3_storage()
        result_data = {
            "raw_text": final_text,
            "confidence": final_conf,
            "engine": engine,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        result_json = json.dumps(result_data, ensure_ascii=False, indent=2)
        file_name = f"ocr_results/ocr_result_{int(time.time())}.json"
        storage.upload_file(
            file_content=result_json.encode('utf-8'),
            file_name=file_name,
            content_type='application/json'
        )
    except Exception as upload_err:
        logger.warning(f"S3上传失败: {upload_err}")

    return OCRRecognizeOutput(
        raw_text=final_text,
        ocr_confidence=final_conf,
        engine_used=engine,
        processing_time=elapsed_time
    )
