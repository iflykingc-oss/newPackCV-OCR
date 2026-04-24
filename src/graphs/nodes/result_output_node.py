# -*- coding: utf-8 -*-
"""
结果输出节点
将处理结果格式化输出，支持多平台推送和文件导出
"""

import os
import json
import time
from typing import Dict, Any
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context
from graphs.state import ResultOutputInput, ResultOutputOutput
from utils.file.file import File, FileOps


def result_output_node(
    state: ResultOutputInput,
    config: RunnableConfig,
    runtime: Runtime[Context]
) -> ResultOutputOutput:
    """
    title: 结果输出
    desc: 将处理结果格式化输出，支持导出为JSON/Excel/PDF，并支持推送到微信或飞书
    integrations: 对象存储, 文档生成
    """
    ctx = runtime.context
    
    try:
        # 处理多种输入字段名（兼容性）
        ocr_result = state.ocr_result or state.raw_text or state.ocr_raw_result or ""
        corrected_text = state.corrected_text or state.corrected_result or ""
        qa_answer = state.qa_answer or state.answer or ""
        image = state.package_image or state.preprocessed_image or None
        
        # 构造最终结果
        final_result = {
            "ocr_result": ocr_result,
            "structured_data": state.structured_data,
            "corrected_text": corrected_text,
            "qa_answer": qa_answer,
            "export_format": state.export_format,
            "platform": state.platform,
            "timestamp": int(time.time())
        }
        
        export_file_url = None
        platform_push_result = {}
        
        # 根据格式导出文件
        if state.export_format == "json":
            export_file_url = _export_json(final_result, ctx)
        elif state.export_format == "excel":
            export_file_url = _export_excel(final_result, ctx)
        elif state.export_format == "pdf":
            export_file_url = _export_pdf(final_result, image, ctx)
        
        # 推送到平台
        if state.platform == "feishu":
            platform_push_result = _push_to_feishu(final_result, export_file_url, ctx)
        elif state.platform == "wechat":
            platform_push_result = _push_to_wechat(final_result, export_file_url, ctx)
        
        print(f"结果输出完成，导出格式: {state.export_format}")
        
        return ResultOutputOutput(
            final_result=final_result,
            export_file_url=export_file_url,
            platform_push_result=platform_push_result
        )
        
    except Exception as e:
        print(f"结果输出失败: {str(e)}")
        return ResultOutputOutput(
            final_result={"error": str(e)},
            export_file_url=None,
            platform_push_result={"error": str(e)}
        )


def _export_json(result: Dict[str, Any], ctx: Context) -> str:
    """导出为JSON文件"""
    try:
        file_path = f"/tmp/ocr_result_{int(time.time())}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        # 上传到对象存储
        from coze_coding_dev_sdk.s3 import S3SyncStorage
        
        storage = S3SyncStorage()
        object_name = f"ocr_results/{os.path.basename(file_path)}"
        url = storage.upload_file(file_path, object_name)
        
        print(f"JSON文件已上传: {url}")
        return url
        
    except Exception as e:
        print(f"JSON导出失败: {str(e)}")
        return ""


def _export_excel(result: Dict[str, Any], ctx: Context) -> str:
    """导出为Excel文件"""
    try:
        import pandas as pd
        
        # 准备数据
        data = []
        
        # OCR结果
        if result.get("ocr_result"):
            data.append({
                "类型": "OCR识别结果",
                "内容": result["ocr_result"]
            })
        
        # 结构化数据
        if result.get("structured_data"):
            for key, value in result["structured_data"].items():
                if key != "error":
                    data.append({
                        "类型": f"结构化数据-{key}",
                        "内容": str(value)
                    })
        
        # 纠错结果
        if result.get("corrected_text"):
            data.append({
                "类型": "纠错结果",
                "内容": result["corrected_text"]
            })
        
        # 问答结果
        if result.get("qa_answer"):
            data.append({
                "类型": "问答答案",
                "内容": result["qa_answer"]
            })
        
        # 创建DataFrame
        df = pd.DataFrame(data)
        
        # 保存到临时文件
        file_path = f"/tmp/ocr_result_{int(time.time())}.xlsx"
        df.to_excel(file_path, index=False, sheet_name="OCR识别结果")
        
        # 上传到对象存储
        from coze_coding_dev_sdk.s3 import S3SyncStorage
        
        storage = S3SyncStorage()
        object_name = f"ocr_results/{os.path.basename(file_path)}"
        url = storage.upload_file(file_path, object_name)
        
        print(f"Excel文件已上传: {url}")
        return url
        
    except Exception as e:
        print(f"Excel导出失败: {str(e)}")
        return ""


def _export_pdf(result: Dict[str, Any], image: File, ctx: Context) -> str:
    """导出为PDF文件"""
    try:
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
        from reportlab.lib import colors
        
        file_path = f"/tmp/ocr_result_{int(time.time())}.pdf"
        doc = SimpleDocTemplate(file_path, pagesize=A4)
        story = []
        styles = getSampleStyleSheet()
        
        # 标题
        title = Paragraph("OCR包装识别报告", styles['Title'])
        story.append(title)
        story.append(Spacer(1, 12))
        
        # 添加图片（如果有）
        if image and image.url:
            try:
                # 如果是URL，先下载
                img_path = image.url
                if img_path.startswith("http://") or img_path.startswith("https://"):
                    import requests
                    temp_img = f"/tmp/report_image_{int(time.time())}.jpg"
                    resp = requests.get(img_path, timeout=10)
                    with open(temp_img, "wb") as f:
                        f.write(resp.content)
                    img_path = temp_img
                
                img = RLImage(img_path, width=4*inch, height=4*inch)
                story.append(img)
                story.append(Spacer(1, 12))
            except Exception as e:
                print(f"添加图片失败: {str(e)}")
        
        # OCR结果
        if result.get("ocr_result"):
            story.append(Paragraph("OCR识别结果:", styles['Heading2']))
            ocr_para = Paragraph(result["ocr_result"].replace('\n', '<br/>'), styles['Normal'])
            story.append(ocr_para)
            story.append(Spacer(1, 12))
        
        # 结构化数据
        if result.get("structured_data"):
            story.append(Paragraph("结构化数据:", styles['Heading2']))
            table_data = [["字段", "值"]]
            for key, value in result["structured_data"].items():
                if key != "error":
                    table_data.append([key, str(value)])
            
            if len(table_data) > 1:
                table = Table(table_data)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(table)
                story.append(Spacer(1, 12))
        
        # 其他结果
        if result.get("corrected_text"):
            story.append(Paragraph("纠错结果:", styles['Heading2']))
            story.append(Paragraph(result["corrected_text"].replace('\n', '<br/>'), styles['Normal']))
            story.append(Spacer(1, 12))
        
        if result.get("qa_answer"):
            story.append(Paragraph("问答答案:", styles['Heading2']))
            story.append(Paragraph(result["qa_answer"].replace('\n', '<br/>'), styles['Normal']))
        
        # 生成PDF
        doc.build(story)
        
        # 上传到对象存储
        from coze_coding_dev_sdk.s3 import S3SyncStorage
        
        storage = S3SyncStorage()
        object_name = f"ocr_results/{os.path.basename(file_path)}"
        url = storage.upload_file(file_path, object_name)
        
        print(f"PDF文件已上传: {url}")
        return url
        
    except Exception as e:
        print(f"PDF导出失败: {str(e)}")
        return ""


def _push_to_feishu(result: Dict[str, Any], file_url: str, ctx: Context) -> Dict[str, Any]:
    """推送到飞书"""
    try:
        # 这里集成飞书推送逻辑
        # 由于用户未配置集成，暂时返回模拟结果
        print("飞书推送功能需配置集成")
        return {
            "platform": "feishu",
            "status": "skipped",
            "message": "需配置飞书集成"
        }
    except Exception as e:
        print(f"飞书推送失败: {str(e)}")
        return {
            "platform": "feishu",
            "status": "error",
            "error": str(e)
        }


def _push_to_wechat(result: Dict[str, Any], file_url: str, ctx: Context) -> Dict[str, Any]:
    """推送到微信"""
    try:
        # 这里集成微信推送逻辑
        # 由于用户未配置集成，暂时返回模拟结果
        print("微信推送功能需配置集成")
        return {
            "platform": "wechat",
            "status": "skipped",
            "message": "需配置微信集成"
        }
    except Exception as e:
        print(f"微信推送失败: {str(e)}")
        return {
            "platform": "wechat",
            "status": "error",
            "error": str(e)
        }
