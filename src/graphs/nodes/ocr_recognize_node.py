# -*- coding: utf-8 -*-
"""
OCR文字识别节点 - 基于Tesseract和内嵌OCR引擎
无需下载外部模型，即插即用
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
    """下载图片并转换为OpenCV格式"""
    try:
        if url.startswith('data:'):
            header, data = url.split(',', 1)
            img_bytes = BytesIO(data.encode())
            pil_img = Image.open(img_bytes)
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            return img
        
        if url.startswith('http://') or url.startswith('https://'):
            response = requests.get(url, timeout=30)
            if response.status_code != 200:
                logger.error(f"下载图片失败: HTTP {response.status_code}")
                return None
            pil_img = Image.open(BytesIO(response.content))
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            return img
        
        if os.path.exists(url):
            img = cv2.imread(url)
            return img
        
        logger.error(f"无效的图片路径: {url}")
        return None
        
    except Exception as e:
        logger.error(f"下载图片异常: {e}")
        return None


class OCRResult:
    """OCR识别结果内部数据结构"""
    def __init__(self, raw_text: str, confidence: float, regions: List, engine: str, metadata: Dict):
        self.raw_text = raw_text
        self.confidence = confidence
        self.regions = regions
        self.engine = engine
        self.metadata = metadata


def use_tesseract_ocr(img: np.ndarray) -> OCRResult:
    """使用Tesseract OCR识别"""
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 高斯锐化
        blur = cv2.GaussianBlur(gray, (0, 0), 3)
        sharpened = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
        
        # CLAHE对比度增强
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(sharpened)
        
        pil_img = Image.fromarray(enhanced)
        
        config = '--psm 6 -l chi_sim+eng --oem 3'
        text = pytesseract.image_to_string(pil_img, config=config)
        data = pytesseract.image_to_data(pil_img, config=config, output_type=pytesseract.Output.DICT)
        
        confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        regions = []
        n_boxes = len(data['text'])
        for i in range(n_boxes):
            if int(data['conf'][i]) > 0:
                regions.append({
                    'text': data['text'][i],
                    'confidence': int(data['conf'][i]),
                    'bbox': {
                        'x': data['left'][i],
                        'y': data['top'][i],
                        'width': data['width'][i],
                        'height': data['height'][i]
                    }
                })
        
        text = text.strip()
        
        result = OCRResult(
            raw_text=text,
            confidence=avg_confidence / 100.0,
            regions=regions,
            engine='tesseract',
            metadata={
                'status': 'success',
                'regions_count': len(regions)
            }
        )
        
        logger.info(f"Tesseract识别成功: {len(text)}字符, 置信度: {avg_confidence:.1f}%")
        return result
        
    except Exception as e:
        logger.error(f"Tesseract OCR错误: {e}")
        return OCRResult(
            raw_text="",
            confidence=0.0,
            regions=[],
            engine='tesseract',
            metadata={'error': str(e)}
        )


def use_builtin_ocr(img: np.ndarray) -> OCRResult:
    """使用内嵌OCR（备用方案）"""
    try:
        from core.ocr.builtin_ocr import builtin_ocr
        
        temp_path = '/tmp/builtin_ocr_input.jpg'
        cv2.imwrite(temp_path, img)
        
        result = builtin_ocr(temp_path)
        
        if result is None:
            return OCRResult(
                raw_text="",
                confidence=0.0,
                regions=[],
                engine='builtin',
                metadata={'error': '引擎返回空结果'}
            )
        
        regions_list = []
        if hasattr(result, 'raw_text') and result.raw_text:
            regions_list.append({
                'text': result.raw_text,
                'confidence': result.confidence if hasattr(result, 'confidence') else 0.5
            })
        
        return OCRResult(
            raw_text=result.raw_text if hasattr(result, 'raw_text') else "",
            confidence=result.confidence if hasattr(result, 'confidence') else 0.0,
            regions=regions_list,
            engine='builtin',
            metadata={'status': 'success'}
        )
        
    except Exception as e:
        logger.error(f"内嵌OCR错误: {e}")
        return OCRResult(
            raw_text="",
            confidence=0.0,
            regions=[],
            engine='builtin',
            metadata={'error': str(e)}
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
    
    logger.info(f"开始OCR识别: {state.package_image.url if state.package_image else 'no image'}")
    
    try:
        # 选择图片：优先使用预处理后的图片
        image_url = ""
        if state.preprocessed_image and state.preprocessed_image.url:
            image_url = state.preprocessed_image.url
        elif state.package_image and state.package_image.url:
            image_url = state.package_image.url
        elif state.image and state.image.url:
            image_url = state.image.url
        else:
            logger.error("无可用图片")
            return OCRRecognizeOutput(
                raw_text="",
                ocr_confidence=0.0,
                engine_used="none",
                processing_time=time.time() - start_time
            )
        
        # 下载图片
        img = download_image(image_url)
        if img is None:
            return OCRRecognizeOutput(
                raw_text="",
                ocr_confidence=0.0,
                engine_used="none",
                processing_time=time.time() - start_time
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
        logger.info(f"识别文本: {result.raw_text[:200] if result.raw_text else '(empty)'}...")
        
        # 上传结果到S3（失败不影响返回结果）
        try:
            storage = _get_s3_storage()
            result_data = {
                "raw_text": result.raw_text,
                "confidence": result.confidence,
                "metadata": result.metadata,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            result_json = json.dumps(result_data, ensure_ascii=False, indent=2)
            result_bytes = result_json.encode('utf-8')
            
            file_name = f"ocr_results/ocr_result_{int(time.time())}.json"
            key = storage.upload_file(
                file_content=result_bytes,
                file_name=file_name,
                content_type='application/json'
            )
            export_url = storage.generate_presigned_url(key=key, expire_time=3600)
            logger.info(f"OCR结果已上传S3: {key}")
        except Exception as upload_err:
            logger.warning(f"S3上传失败，不影响结果返回: {upload_err}")
        
        return OCRRecognizeOutput(
            ocr_raw_result=result.raw_text,
            raw_text=result.raw_text,
            ocr_confidence=result.confidence,
            confidence=result.confidence,
            ocr_regions=result.regions,
            regions=result.regions,
            engine_used=result.engine,
            processing_time=elapsed_time
        )
        
    except Exception as e:
        logger.error(f"OCR识别异常: {e}")
        return OCRRecognizeOutput(
            ocr_raw_result="",
            raw_text="",
            ocr_confidence=0.0,
            confidence=0.0,
            engine_used="error",
            processing_time=time.time() - start_time
        )
