# -*- coding: utf-8 -*-
"""
并行处理引擎节点
对多个ROI区域进行并行OCR识别处理
"""

import os
import traceback
from typing import Dict, Any, List, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context

from graphs.state import (
    ParallelProcessingInput,
    ParallelProcessingOutput
)


def parallel_processing_node(state: ParallelProcessingInput, config: RunnableConfig, runtime: Runtime[Context]) -> ParallelProcessingOutput:
    """
    title: 并行处理引擎
    desc: 对多个ROI区域进行并行OCR识别处理
    """
    ctx = runtime.context
    
    print(f"[并行处理] 开始处理 {len(state.roi_images)} 个ROI区域...")
    
    try:
        start_time = datetime.now()
        
        # 限制最大并行数
        max_workers = min(state.max_workers, len(state.roi_images))
        
        # 初始化OCR引擎
        ocr_engine = initialize_ocr_engine(state.ocr_engine_type, state.ocr_api_config)
        
        # 使用线程池并行处理
        processing_results = []
        success_count = 0
        failed_count = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_region = {
                executor.submit(
                    process_roi,
                    roi_item,
                    state.roi_regions[i] if i < len(state.roi_regions) else {},
                    ocr_engine,
                    state.ocr_engine_type,
                    state.ocr_api_config,
                    state.enable_expiry_detection,
                    i
                ): i
                for i, roi_item in enumerate(state.roi_images)
            }
            
            # 收集结果
            for future in as_completed(future_to_region):
                region_idx = future_to_region[future]
                try:
                    result = future.result()
                    processing_results.append(result)
                    if result.get('status') == 'success':
                        success_count += 1
                    else:
                        failed_count += 1
                    
                    print(f"[并行处理] 区域 {region_idx + 1} 完成: {result.get('status')}")
                except Exception as e:
                    error_msg = f"区域 {region_idx + 1} 处理失败: {str(e)}"
                    print(f"[并行处理] {error_msg}")
                    processing_results.append({
                        'region_index': region_idx,
                        'status': 'failed',
                        'error_message': error_msg
                    })
                    failed_count += 1
        
        # 按原始顺序排序结果
        processing_results.sort(key=lambda x: x.get('region_index', 0))
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        print(f"[并行处理] 完成，成功: {success_count}, 失败: {failed_count}, 耗时: {processing_time:.2f}秒")
        
        return ParallelProcessingOutput(
            processing_results=processing_results,
            total_processed=len(processing_results),
            success_count=success_count,
            failed_count=failed_count,
            processing_time=processing_time
        )
        
    except Exception as e:
        error_msg = f"并行处理引擎发生错误: {str(e)}\n{traceback.format_exc()}"
        print(f"[并行处理] 错误: {error_msg}")
        
        return ParallelProcessingOutput(
            processing_results=[],
            total_processed=0,
            success_count=0,
            failed_count=len(state.roi_images),
            processing_time=0.0
        )


def initialize_ocr_engine(ocr_engine_type: str, ocr_api_config: Optional[Dict[str, Any]]):
    """初始化OCR引擎"""
    try:
        if ocr_engine_type == "builtin":
            import paddleocr
            # 初始化PaddleOCR
            engine = paddleocr.PaddleOCR(use_angle_cls=True, lang="ch")
            return engine
        elif ocr_engine_type == "tesseract":
            # Tesseract不需要预初始化
            return "tesseract"
        else:
            return None
    except Exception as e:
        print(f"初始化OCR引擎失败: {str(e)}，使用Tesseract作为备选")
        return "tesseract"


def process_roi(
    roi_item: Dict[str, Any],
    region_info: Dict[str, Any],
    ocr_engine,
    ocr_engine_type: str,
    ocr_api_config: Optional[Dict[str, Any]],
    enable_expiry_detection: bool,
    region_index: int
) -> Dict[str, Any]:
    """处理单个ROI区域"""
    try:
        roi_url = roi_item.get('url')
        if not roi_url:
            return {
                'region_index': region_index,
                'status': 'failed',
                'error_message': 'ROI URL为空'
            }
        
        # 下载ROI图片
        roi_image_data = download_image(roi_url)
        if not roi_image_data:
            raise Exception("下载ROI图片失败")
        
        # 解码图片
        import cv2
        import numpy as np
        img_array = np.frombuffer(roi_image_data, np.uint8)
        roi_image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if roi_image is None:
            raise Exception("ROI图片解码失败")
        
        # OCR识别
        ocr_result = perform_ocr(roi_image, ocr_engine, ocr_engine_type, ocr_api_config)
        
        # 效期检测（如果启用）
        expiry_info = None
        if enable_expiry_detection and ocr_result.get('text'):
            expiry_info = detect_expiry_date(ocr_result['text'])
        
        return {
            'region_index': region_index,
            'status': 'success',
            'roi_url': roi_url,
            'region_info': region_info,
            'ocr_result': ocr_result,
            'expiry_info': expiry_info,
            'processing_time': ocr_result.get('processing_time', 0.0)
        }
        
    except Exception as e:
        return {
            'region_index': region_index,
            'status': 'failed',
            'error_message': str(e)
        }


def download_image(image_url: str) -> Optional[bytes]:
    """下载图片"""
    try:
        import requests
        response = requests.get(image_url, timeout=30)
        if response.status_code == 200:
            return response.content
        return None
    except Exception as e:
        print(f"下载图片失败: {str(e)}")
        return None


def perform_ocr(image, ocr_engine, ocr_engine_type: str, ocr_api_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """执行OCR识别"""
    from datetime import datetime
    start_time = datetime.now()
    
    try:
        if ocr_engine_type == "builtin" and ocr_engine != "tesseract":
            # 使用PaddleOCR
            result = ocr_engine.ocr(image, cls=True)
            
            if not result or not result[0]:
                return {
                    'text': '',
                    'confidence': 0.0,
                    'regions': [],
                    'processing_time': (datetime.now() - start_time).total_seconds()
                }
            
            # 提取文本
            texts = []
            regions = []
            total_confidence = 0.0
            
            for line in result[0]:
                if line:
                    text = line[1][0]
                    confidence = line[1][1]
                    texts.append(text)
                    regions.append({
                        'text': text,
                        'confidence': confidence,
                        'bbox': line[0]
                    })
                    total_confidence += confidence
            
            full_text = "\n".join(texts)
            avg_confidence = total_confidence / len(texts) if texts else 0.0
            
            return {
                'text': full_text,
                'confidence': avg_confidence,
                'regions': regions,
                'processing_time': (datetime.now() - start_time).total_seconds()
            }
        else:
            # 使用Tesseract
            import pytesseract
            text = pytesseract.image_to_string(image, lang='chi_sim+eng')
            return {
                'text': text.strip(),
                'confidence': 0.8,
                'regions': [],
                'processing_time': (datetime.now() - start_time).total_seconds()
            }
    except Exception as e:
        print(f"OCR识别失败: {str(e)}")
        return {
            'text': '',
            'confidence': 0.0,
            'regions': [],
            'processing_time': (datetime.now() - start_time).total_seconds()
        }


def detect_expiry_date(text: str) -> Optional[Dict[str, Any]]:
    """检测文本中的生产日期和有效期"""
    import re
    from datetime import datetime, timedelta
    
    expiry_info = {
        'production_date': None,
        'expiry_date': None,
        'shelf_life_days': None,
        'days_to_expiry': None,
        'status': 'unknown'
    }
    
    # 日期格式正则表达式（支持多种格式）
    date_patterns = [
        r'(\d{4})[-/年](\d{1,2})[-/月](\d{1,2})[日]?',  # 2024-01-15 或 2024年1月15日
        r'(\d{4})(\d{2})(\d{2})',  # 20240115
        r'生产日期[:：](\d{4})[-/年](\d{1,2})[-/月](\d{1,2})[日]?',  # 生产日期：2024-01-15
        r'有效期至[:：](\d{4})[-/年](\d{1,2})[-/月](\d{1,2})[日]?',  # 有效期至：2024-01-15
        r'保质期[:：](\d+)[天月年]',  # 保质期：18个月
    ]
    
    # 查找日期
    dates_found = []
    for pattern in date_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            try:
                if len(match) == 3:
                    year, month, day = int(match[0]), int(match[1]), int(match[2])
                    dates_found.append(datetime(year, month, day))
            except:
                continue
    
    if dates_found:
        # 假设第一个日期是生产日期，第二个是有效期
        if len(dates_found) >= 2:
            expiry_info['production_date'] = dates_found[0].strftime('%Y-%m-%d')
            expiry_info['expiry_date'] = dates_found[1].strftime('%Y-%m-%d')
        else:
            expiry_info['expiry_date'] = dates_found[0].strftime('%Y-%m-%d')
        
        # 计算天数
        if expiry_info['expiry_date']:
            try:
                expiry_date = datetime.strptime(expiry_info['expiry_date'], '%Y-%m-%d')
                today = datetime.now()
                days_to_expiry = (expiry_date - today).days
                
                if days_to_expiry < 0:
                    expiry_info['status'] = 'expired'
                elif days_to_expiry <= 30:
                    expiry_info['status'] = 'near_expiry'
                else:
                    expiry_info['status'] = 'valid'
                
                expiry_info['days_to_expiry'] = days_to_expiry
            except:
                pass
    
    return expiry_info if any(expiry_info.values()) else None
