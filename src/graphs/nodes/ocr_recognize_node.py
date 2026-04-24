# -*- coding: utf-8 -*-
"""
OCR识别节点
支持内置OCR算法和外部API调用，识别包装上的文字信息
"""

import os
import json
import time
import requests
from typing import Dict, Any, List
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context
from graphs.state import OCRRecognizeInput, OCRRecognizeOutput
from utils.file.file import File


def ocr_recognize_node(
    state: OCRRecognizeInput,
    config: RunnableConfig,
    runtime: Runtime[Context]
) -> OCRRecognizeOutput:
    """
    title: OCR识别
    desc: 使用内置OCR算法或外部API识别包装上的文字信息，支持多语言混合文本
    integrations: 
    """
    ctx = runtime.context
    start_time = time.time()
    
    # 获取图片路径（优先使用预处理图片，然后是原始图片）
    if state.image:
        image_path = state.image.url
    elif state.preprocessed_image:
        image_path = state.preprocessed_image.url
    elif state.package_image:
        image_path = state.package_image.url
    else:
        print("没有可用的图片")
        return OCRRecognizeOutput(
            ocr_raw_result="",
            raw_text="",
            ocr_confidence=0.0,
            confidence=0.0,
            ocr_regions=[],
            regions=[],
            engine_used="none",
            processing_time=time.time() - start_time
        )
    
    # 如果是URL，先下载
    if image_path.startswith("http://") or image_path.startswith("https://"):
        temp_path = f"/tmp/ocr_input_{int(time.time())}.jpg"
        try:
            resp = requests.get(image_path, timeout=30)
            resp.raise_for_status()
            with open(temp_path, "wb") as f:
                f.write(resp.content)
            image_path = temp_path
        except Exception as e:
            print(f"下载图片失败: {str(e)}")
            return OCRRecognizeOutput(
                ocr_raw_result="",
                raw_text="",
                ocr_confidence=0.0,
                confidence=0.0,
                ocr_regions=[],
                regions=[],
                engine_used="none",
                processing_time=time.time() - start_time
            )
    
    # 根据OCR引擎类型选择识别方式
    if state.ocr_engine_type == "api" and state.ocr_api_config:
        return _call_ocr_api(state, image_path, start_time, ctx)
    else:
        return _use_builtin_ocr(state, image_path, start_time, ctx)


def _use_builtin_ocr(
    state: OCRRecognizeInput,
    image_path: str,
    start_time: float,
    ctx: Context
) -> OCRRecognizeOutput:
    """使用内置OCR算法（PaddleOCR或EasyOCR）"""
    try:
        # 尝试使用PaddleOCR
        from paddleocr import PaddleOCR
        
        print("使用内置PaddleOCR引擎进行识别")
        
        # 初始化OCR引擎（支持中英文混合）
        ocr = PaddleOCR(
            use_angle_cls=True,
            lang="ch",
            use_gpu=False,
            show_log=False
        )
        
        # 执行识别
        result = ocr.ocr(image_path, cls=True)
        
        # 解析结果
        raw_text = ""
        regions = []
        total_confidence = 0.0
        region_count = 0
        
        if result and result[0]:
            for line in result[0]:
                if line:
                    # line格式: [[[x1,y1], [x2,y2], [x3,y3], [x4,y4]], (text, confidence)]
                    box = line[0]
                    text_info = line[1]
                    
                    text = text_info[0] if text_info else ""
                    conf = float(text_info[1]) if text_info and len(text_info) > 1 else 0.0
                    
                    if text:
                        raw_text += text + "\n"
                        regions.append({
                            "text": text,
                            "confidence": conf,
                            "bbox": box
                        })
                        total_confidence += conf
                        region_count += 1
        
        # 计算平均置信度
        avg_confidence = total_confidence / region_count if region_count > 0 else 0.0
        
        processing_time = time.time() - start_time
        
        print(f"OCR识别完成，耗时{processing_time:.2f}秒，识别{len(regions)}个区域")
        
        # 确保输出两种字段名（ocr_raw_result 和 raw_text），兼容不同的下游节点
        return OCRRecognizeOutput(
            ocr_raw_result=raw_text.strip(),
            raw_text=raw_text.strip(),
            ocr_confidence=avg_confidence,
            confidence=avg_confidence,
            ocr_regions=regions,
            regions=regions,
            engine_used="paddleocr",
            processing_time=processing_time
        )
        
    except ImportError:
        print("PaddleOCR未安装，尝试使用Tesseract OCR")
        return _use_tesseract_ocr(image_path, start_time, ctx)
    except Exception as e:
        print(f"PaddleOCR识别失败: {str(e)}")
        return OCRRecognizeOutput(
            ocr_raw_result="",
            raw_text="",
            ocr_confidence=0.0,
            confidence=0.0,
            ocr_regions=[],
            regions=[],
            engine_used="none",
            processing_time=time.time() - start_time
        )


def _use_tesseract_ocr(image_path: str, start_time: float, ctx: Context) -> OCRRecognizeOutput:
    """使用Tesseract OCR作为备选方案"""
    try:
        import pytesseract
        
        # 识别文本
        text = pytesseract.image_to_string(image_path, lang='chi_sim+eng')
        
        # 获取详细数据（包括置信度）
        data = pytesseract.image_to_data(image_path, lang='chi_sim+eng', output_type=pytesseract.Output.DICT)
        
        # 解析区域
        regions = []
        confidences = []
        for i in range(len(data['text'])):
            text = data['text'][i]
            conf = data['conf'][i]
            
            if text.strip() and int(conf) > 0:
                regions.append({
                    "text": text,
                    "confidence": int(conf) / 100.0,
                    "bbox": [
                        [data['left'][i], data['top'][i]],
                        [data['left'][i] + data['width'][i], data['top'][i]],
                        [data['left'][i] + data['width'][i], data['top'][i] + data['height'][i]],
                        [data['left'][i], data['top'][i] + data['height'][i]]
                    ]
                })
                confidences.append(int(conf) / 100.0)
        
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return OCRRecognizeOutput(
            ocr_raw_result=text,
            raw_text=text,
            ocr_confidence=avg_confidence,
            confidence=avg_confidence,
            ocr_regions=regions,
            regions=regions,
            engine_used="tesseract",
            processing_time=time.time() - start_time
        )
        
    except Exception as e:
        print(f"Tesseract OCR识别失败: {str(e)}")
        return OCRRecognizeOutput(
            ocr_raw_result="",
            raw_text="",
            ocr_confidence=0.0,
            confidence=0.0,
            ocr_regions=[],
            regions=[],
            engine_used="none",
            processing_time=time.time() - start_time
        )


def _call_ocr_api(
    state: OCRRecognizeInput,
    image_path: str,
    start_time: float,
    ctx: Context
) -> OCRRecognizeOutput:
    """调用外部OCR API"""
    try:
        api_config = state.ocr_api_config
        api_url = api_config.get("url", "")
        api_key = api_config.get("api_key", "")
        
        if not api_url:
            print("OCR API配置缺少url")
            return OCRRecognizeOutput(
                ocr_raw_result="",
                raw_text="",
                ocr_confidence=0.0,
                confidence=0.0,
                ocr_regions=[],
                regions=[],
                engine_used="api_failed",
                processing_time=time.time() - start_time
            )
        
        # 读取图片并转换为base64
        with open(image_path, "rb") as f:
            image_data = f.read()
        import base64
        image_base64 = base64.b64encode(image_data).decode("utf-8")
        
        # 调用API（示例格式，实际根据API文档调整）
        headers = {
            "Content-Type": "application/json"
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        payload = {
            "image": image_base64,
            "language": "zh-cn,en"
        }
        
        resp = requests.post(api_url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        
        result = resp.json()
        
        # 解析API响应（根据实际API格式调整）
        raw_text = result.get("text", "")
        regions = result.get("regions", [])
        confidence = result.get("confidence", 0.0)
        
        return OCRRecognizeOutput(
            ocr_raw_result=raw_text,
            raw_text=raw_text,
            ocr_confidence=confidence,
            confidence=confidence,
            ocr_regions=regions,
            regions=regions,
            engine_used="external_api",
            processing_time=time.time() - start_time
        )
        
    except Exception as e:
        print(f"OCR API调用失败: {str(e)}")
        # API失败时，回退到内置OCR
        return _use_builtin_ocr(state, image_path, start_time, ctx)
