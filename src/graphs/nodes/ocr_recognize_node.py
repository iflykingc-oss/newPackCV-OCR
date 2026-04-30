"""
OCR识别节点 - 基于内嵌OCR + Tesseract
不依赖外部API下载，即插即用
"""

import io
import json
import base64
import time
import logging
from typing import Optional, List, Dict, Any

import cv2
import numpy as np
import pytesseract
from PIL import Image
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context

from graphs.state import (
    OCRRecognizeInput,
    OCRRecognizeOutput,
    OCRResult
)
from storage.oss import OSSStorage
from core.ocr.builtin_ocr import builtin_ocr, pattern_ocr, extract_structured_info


logger = logging.getLogger(__name__)


def download_image(url: str) -> Optional[np.ndarray]:
    """下载图片"""
    import requests
    
    try:
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            img_array = np.frombuffer(response.content, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            return img
    except Exception as e:
        logger.warning(f"下载图片失败: {e}")
    return None


def preprocess_for_ocr(img: np.ndarray) -> np.ndarray:
    """OCR专用图像预处理"""
    # 转为灰度
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # 去噪
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    
    # 对比度增强 (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    
    # 锐化
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    return sharpened


def use_tesseract_ocr(img: np.ndarray) -> OCRResult:
    """
    使用Tesseract OCR进行文字识别
    支持中英文混合识别
    """
    try:
        # 预处理
        processed = preprocess_for_ocr(img)
        
        # 转为PIL Image
        pil_img = Image.fromarray(processed)
        
        # Tesseract配置：中文+英文，数字优先
        config = '--psm 6 -l chi_sim+eng --oem 3'
        
        # 识别文字
        text = pytesseract.image_to_string(pil_img, config=config)
        
        # 获取置信度
        data = pytesseract.image_to_data(pil_img, config=config, output_type=pytesseract.Output.DICT)
        
        # 计算平均置信度
        confidences = [conf for conf in data['conf'] if conf != -1]
        avg_confidence = sum(confidences) / len(confidences) / 100.0 if confidences else 0.5
        
        # 清理文本
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        cleaned_text = '\n'.join(lines)
        
        # 提取结构化信息
        structured = extract_structured_info(cleaned_text)
        
        return OCRResult(
            raw_text=cleaned_text,
            confidence=avg_confidence,
            regions=[],
            metadata={
                "engine": "tesseract",
                "languages": ["chi_sim", "eng"],
                "structured_info": structured
            }
        )
        
    except Exception as e:
        logger.error(f"Tesseract OCR错误: {e}")
        return OCRResult(
            raw_text="",
            confidence=0.0,
            regions=[],
            metadata={"engine": "tesseract", "error": str(e)}
        )


def use_builtin_ocr(img: np.ndarray) -> OCRResult:
    """
    使用内嵌的轻量级OCR
    基于OpenCV图像处理
    """
    try:
        # 保存到临时文件
        _, buffer = cv2.imencode('.jpg', img)
        temp_path = '/tmp/builtin_ocr_temp.jpg'
        cv2.imwrite(temp_path, img)
        
        # 使用内嵌OCR
        result = builtin_ocr.ocr(temp_path)
        
        return OCRResult(
            raw_text=result.raw_text,
            confidence=result.confidence,
            regions=result.regions,
            metadata={"engine": "builtin", **result.metadata}
        )
        
    except Exception as e:
        logger.error(f"内嵌OCR错误: {e}")
        return OCRResult(
            raw_text="",
            confidence=0.0,
            regions=[],
            metadata={"engine": "builtin", "error": str(e)}
        )


def ocr_recognize_node(
    state: OCRRecognizeInput,
    config: RunnableConfig,
    runtime: Runtime[Context]
) -> OCRRecognizeOutput:
    """
    title: OCR文字识别
    desc: 识别图片中的文字内容，支持中文和英文。使用Tesseract OCR引擎，无需下载外部模型。
    integrations: Tesseract OCR
    """
    ctx = runtime.context
    start_time = time.time()
    
    logger.info(f"开始OCR识别: {state.package_image.url}")
    
    try:
        # 下载图片
        img = download_image(state.package_image.url)
        if img is None:
            return OCRRecognizeOutput(
                raw_text="",
                ocr_result=OCRResult(
                    raw_text="",
                    confidence=0.0,
                    regions=[],
                    metadata={"error": "无法下载图片"}
                ),
                export_file_url=""
            )
        
        logger.info(f"图片下载成功，尺寸: {img.shape}")
        
        # 优先使用Tesseract OCR
        result = use_tesseract_ocr(img)
        
        # 如果Tesseract没有结果，尝试内嵌OCR
        if not result.raw_text.strip():
            logger.info("Tesseract无结果，尝试内嵌OCR")
            result = use_builtin_ocr(img)
        
        elapsed_time = time.time() - start_time
        logger.info(f"OCR识别完成，耗时: {elapsed_time:.2f}秒")
        logger.info(f"识别文本: {result.raw_text[:200]}...")
        
        # 上传结果到OSS
        oss = OSSStorage()
        result_data = {
            "raw_text": result.raw_text,
            "confidence": result.confidence,
            "metadata": result.metadata,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        result_json = json.dumps(result_data, ensure_ascii=False, indent=2)
        result_bytes = result_json.encode('utf-8')
        
        file_name = f"ocr_results/ocr_result_{int(time.time())}.json"
        oss.upload_file(result_bytes, file_name, 'application/json')
        export_url = oss.generate_presigned_url(file_name, expire_time=3600)
        
        return OCRRecognizeOutput(
            raw_text=result.raw_text,
            ocr_result=result,
            export_file_url=export_url
        )
        
    except Exception as e:
        logger.error(f"OCR识别异常: {e}")
        return OCRRecognizeOutput(
            raw_text="",
            ocr_result=OCRResult(
                raw_text="",
                confidence=0.0,
                regions=[],
                metadata={"error": str(e)}
            ),
            export_file_url=""
        )
