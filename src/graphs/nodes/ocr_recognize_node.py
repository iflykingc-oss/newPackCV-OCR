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
from pydantic import BaseModel, Field

from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context
from utils.file.file import File
from storage.oss import OSSStorage
from core.ocr.builtin_ocr import builtin_ocr

logger = logging.getLogger(__name__)


def download_image(url: str) -> Optional[np.ndarray]:
    """下载图片并转换为OpenCV格式"""
    try:
        if url.startswith('data:'):
            # Base64编码的图片
            header, data = url.split(',', 1)
            img_bytes = BytesIO(data.encode())
            pil_img = Image.open(img_bytes)
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            return img
        
        if url.startswith('http://') or url.startswith('https://'):
            # 网络图片
            response = requests.get(url, timeout=30)
            if response.status_code != 200:
                logger.error(f"下载图片失败: HTTP {response.status_code}")
                return None
            pil_img = Image.open(BytesIO(response.content))
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            return img
        
        # 本地文件
        if os.path.exists(url):
            img = cv2.imread(url)
            return img
        
        logger.error(f"无效的图片路径: {url}")
        return None
        
    except Exception as e:
        logger.error(f"下载图片异常: {e}")
        return None


class OCRResult:
    """OCR识别结果"""
    def __init__(self, raw_text: str, confidence: float, regions: List, engine: str, metadata: Dict):
        self.raw_text = raw_text
        self.confidence = confidence
        self.regions = regions
        self.engine = engine
        self.metadata = metadata


def use_tesseract_ocr(img: np.ndarray) -> OCRResult:
    """使用Tesseract OCR识别"""
    try:
        # 图像预处理
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 自适应阈值增强
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
        
        # CLAHE增强
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(thresh)
        
        # 形态学操作去噪
        kernel = np.ones((1, 1), np.uint8)
        processed = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
        
        pil_img = Image.fromarray(processed)
        
        # Tesseract配置：中文+英文
        config = '--psm 6 -l chi_sim+eng --oem 3'
        
        # 识别文字
        text = pytesseract.image_to_string(pil_img, config=config)
        
        # 获取置信度
        data = pytesseract.image_to_data(pil_img, config=config, output_type=pytesseract.Output.DICT)
        
        # 计算平均置信度
        confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        # 构建regions
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
        
        # 清理文本
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
        # 保存临时文件
        temp_path = '/tmp/builtin_ocr_input.jpg'
        cv2.imwrite(temp_path, img)
        
        # 使用内嵌OCR
        result = builtin_ocr(temp_path)
        
        # 检查结果是否有效
        if result is None:
            return OCRResult(
                raw_text="",
                confidence=0.0,
                regions=[],
                engine='builtin',
                metadata={'error': '引擎返回空结果'}
            )
        
        # 构建regions
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


class OCRRecognizeInput(BaseModel):
    """OCR识别输入"""
    package_image: File = Field(..., description="待识别的图片")
    preprocessed_image: Optional[File] = Field(default=None, description="预处理后的图片")
    is_enhanced: bool = Field(default=False, description="是否已预处理增强")
    ocr_engine_type: str = Field(default="builtin", description="OCR引擎类型")


class OCRRecognizeOutput(BaseModel):
    """OCR识别输出"""
    ocr_raw_result: str = Field(default="", description="OCR原始识别结果")
    raw_text: str = Field(default="", description="识别出的文本")
    ocr_confidence: float = Field(default=0.0, description="OCR置信度")
    confidence: float = Field(default=0.0, description="整体置信度")
    ocr_regions: List = Field(default_factory=list, description="OCR区域列表")
    regions: List = Field(default_factory=list, description="文本区域列表")
    engine_used: str = Field(default="", description="使用的OCR引擎")
    processing_time: float = Field(default=0.0, description="处理耗时(秒)")


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
        
        # 上传结果到OSS（失败不影响返回结果）
        export_url = ""
        try:
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
        except Exception as upload_err:
            logger.warning(f"OSS上传失败，不影响结果返回: {upload_err}")
        
        return OCRRecognizeOutput(
            raw_text=result.raw_text,
            ocr_confidence=result.confidence,
            ocr_regions=result.regions,
            regions=result.regions,
            engine_used=result.engine,
            processing_time=elapsed_time
        )
        
    except Exception as e:
        logger.error(f"OCR识别异常: {e}")
        return OCRRecognizeOutput(
            raw_text="",
            ocr_confidence=0.0,
            engine_used="error",
            processing_time=time.time() - start_time
        )
