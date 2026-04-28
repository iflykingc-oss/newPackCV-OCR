# -*- coding: utf-8 -*-
"""
批量处理节点
支持多张图片的批量OCR识别和结果处理
"""

import os
import json
import traceback
import requests
import tempfile
from typing import Dict, Any, Optional, List
from datetime import datetime
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context

from graphs.state import (
    BatchProcessInput,
    BatchProcessOutput,
    GlobalState
)


def batch_process_node(state: BatchProcessInput, config: RunnableConfig, runtime: Runtime[Context]) -> BatchProcessOutput:
    """
    title: 批量处理
    desc: 对多张图片进行批量OCR识别和结果处理
    """
    ctx = runtime.context
    
    print(f"[批量处理] 开始处理 {len(state.images)} 张图片")
    
    try:
        # 导入依赖（延迟导入避免循环依赖）
        import cv2
        import numpy as np
        import paddleocr
        import pytesseract
        from io import BytesIO
        
        # 导入对象存储
        from coze_coding_dev_sdk.s3 import S3SyncStorage
        
        # 初始化OCR引擎
        ocr_engine = None
        try:
            ocr_engine = paddleocr.PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
        except Exception as e:
            print(f"[批量处理] 初始化PaddleOCR失败: {str(e)}")
        
        # 处理每张图片
        results = []
        success_count = 0
        failed_count = 0
        errors = []
        
        for idx, image_file in enumerate(state.images):
            image_url = image_file.url
            print(f"[批量处理] 处理第 {idx + 1}/{len(state.images)} 张图片: {image_url}")
            
            try:
                # 1. 下载图片
                img_data = download_image(image_url)
                if img_data is None:
                    raise Exception("图片下载失败")
                
                # 2. 解码图片
                img_array = np.frombuffer(img_data, np.uint8)
                image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                if image is None:
                    raise Exception("图片解码失败")
                
                # 3. 图像预处理（简化版）
                preprocessed = preprocess_image(image)
                
                # 4. OCR识别
                ocr_engine_type = state.ocr_engine_type if state.ocr_engine_type else "builtin"
                ocr_result = {}
                
                if ocr_engine_type == "builtin" and ocr_engine:
                    # 使用PaddleOCR
                    ocr_result = recognize_with_paddleocr(ocr_engine, preprocessed)
                elif ocr_engine_type == "tesseract":
                    # 使用Tesseract
                    ocr_result = recognize_with_tesseract(preprocessed)
                else:
                    # 默认使用PaddleOCR
                    if ocr_engine:
                        ocr_result = recognize_with_paddleocr(ocr_engine, preprocessed)
                    else:
                        ocr_result = recognize_with_tesseract(preprocessed)
                
                if not ocr_result or not ocr_result.get("text"):
                    raise Exception("OCR识别失败或无结果")
                
                # 5. 构建结果
                result_item = {
                    "image_index": idx + 1,
                    "image_url": image_url,
                    "status": "success",
                    "ocr_text": ocr_result.get("text", ""),
                    "ocr_raw_result": ocr_result.get("text", ""),  # 兼容字段
                    "confidence": ocr_result.get("confidence", 0.0),
                    "regions": ocr_result.get("regions", [])
                }
                
                results.append(result_item)
                success_count += 1
                print(f"[批量处理] ✓ 第 {idx + 1} 张图片识别成功")
                
            except Exception as e:
                error_msg = f"第 {idx + 1} 张图片处理失败: {str(e)}"
                print(f"[批量处理] ✗ {error_msg}")
                errors.append(error_msg)
                
                result_item = {
                    "image_index": idx + 1,
                    "image_url": image_url,
                    "status": "failed",
                    "error_message": str(e)
                }
                
                results.append(result_item)
                failed_count += 1
        
        # 6. 合并结果
        all_text = "\n\n".join([
            f"图片 {r['image_index']}: {r.get('ocr_text', '')}"
            for r in results if r.get("status") == "success"
        ])
        
        # 7. 导出结果
        export_file_url = None
        export_format = state.export_format if state.export_format else "json"
        
        if export_format == "json":
            # JSON结果直接返回，不导出
            export_file_url = None
        elif export_format == "excel":
            export_file_url = export_batch_to_excel(results)
        elif export_format == "pdf":
            export_file_url = export_batch_to_pdf(results)
        
        print(f"[批量处理] 完成，成功: {success_count}, 失败: {failed_count}")
        
        return BatchProcessOutput(
            results=results,
            all_text=all_text,
            success_count=success_count,
            failed_count=failed_count,
            errors=errors,
            export_file_url=export_file_url
        )
        
    except Exception as e:
        error_msg = f"批量处理节点发生错误: {str(e)}\n{traceback.format_exc()}"
        print(f"[批量处理] 错误: {error_msg}")
        
        return BatchProcessOutput(
            results=[],
            all_text="",
            success_count=0,
            failed_count=len(state.images),
            errors=[error_msg],
            export_file_url=None
        )


def download_image(image_url: str) -> Optional[bytes]:
    """下载图片"""
    try:
        response = requests.get(image_url, timeout=30)
        if response.status_code == 200:
            return response.content
        return None
    except Exception as e:
        print(f"下载图片失败: {str(e)}")
        return None


def preprocess_image(image):
    """图像预处理（简化版）"""
    import cv2
    
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 自适应阈值处理
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )
    
    # 去噪
    denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)
    
    return denoised


def recognize_with_paddleocr(ocr_engine, image):
    """使用PaddleOCR识别"""
    try:
        result = ocr_engine.ocr(image, cls=True)
        
        if not result or not result[0]:
            return {"text": "", "confidence": 0.0, "regions": []}
        
        # 提取文本和置信度
        texts = []
        regions = []
        total_confidence = 0.0
        
        for line in result[0]:
            if line:
                text = line[1][0]
                confidence = line[1][1]
                texts.append(text)
                regions.append({
                    "text": text,
                    "confidence": confidence,
                    "bbox": line[0]
                })
                total_confidence += confidence
        
        full_text = "\n".join(texts)
        avg_confidence = total_confidence / len(texts) if texts else 0.0
        
        return {
            "text": full_text,
            "confidence": avg_confidence,
            "regions": regions
        }
    except Exception as e:
        print(f"PaddleOCR识别失败: {str(e)}")
        return {"text": "", "confidence": 0.0, "regions": []}


def recognize_with_tesseract(image):
    """使用Tesseract识别"""
    try:
        import pytesseract
        text = pytesseract.image_to_string(image, lang='chi_sim+eng')
        return {
            "text": text.strip(),
            "confidence": 0.8,  # Tesseract不提供置信度，使用默认值
            "regions": []
        }
    except Exception as e:
        print(f"Tesseract识别失败: {str(e)}")
        return {"text": "", "confidence": 0.0, "regions": []}


def export_batch_to_excel(results: list) -> str:
    """导出批量结果到Excel"""
    try:
        import pandas as pd
        from coze_coding_dev_sdk.s3 import S3SyncStorage
        
        # 准备数据
        data = []
        for r in results:
            data.append({
                "图片序号": r.get("image_index", ""),
                "图片URL": r.get("image_url", ""),
                "状态": r.get("status", ""),
                "识别文本": r.get("ocr_text", ""),
                "置信度": r.get("confidence", 0.0),
                "错误信息": r.get("error_message", "")
            })
        
        # 创建DataFrame
        df = pd.DataFrame(data)
        
        # 保存到临时文件
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.xlsx', delete=False) as f:
            temp_path = f.name
            df.to_excel(f, index=False, engine='openpyxl')
        
        # 上传到对象存储
        storage = S3SyncStorage()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        object_name = f"ocr/batch_results_{timestamp}.xlsx"
        url = storage.upload_file(temp_path, object_name)
        
        # 删除临时文件
        try:
            os.unlink(temp_path)
        except:
            pass
        
        print(f"[批量处理] Excel文件已上传: {url}")
        return url
        
    except Exception as e:
        print(f"导出Excel失败: {str(e)}")
        return None


def export_batch_to_pdf(results: list) -> str:
    """导出批量结果到PDF"""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.units import cm
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
        from coze_coding_dev_sdk.s3 import S3SyncStorage
        
        # 保存到临时文件
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pdf', delete=False) as f:
            temp_path = f.name
            
            doc = SimpleDocTemplate(f, pagesize=A4)
            
            # 准备数据
            table_data = [["序号", "图片URL", "状态", "识别文本", "置信度", "错误信息"]]
            for r in results:
                table_data.append([
                    str(r.get("image_index", "")),
                    r.get("image_url", ""),
                    r.get("status", ""),
                    r.get("ocr_text", ""),
                    str(r.get("confidence", 0.0)),
                    r.get("error_message", "")
                ])
            
            # 创建表格
            table = Table(table_data, colWidths=[1*cm, 4*cm, 2*cm, 6*cm, 2*cm, 3*cm])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            
            # 构建PDF文档
            doc.build([table])
        
        # 上传到对象存储
        storage = S3SyncStorage()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        object_name = f"ocr/batch_results_{timestamp}.pdf"
        url = storage.upload_file(temp_path, object_name)
        
        # 删除临时文件
        try:
            os.unlink(temp_path)
        except:
            pass
        
        print(f"[批量处理] PDF文件已上传: {url}")
        return url
        
    except Exception as e:
        print(f"导出PDF失败: {str(e)}")
        return None
