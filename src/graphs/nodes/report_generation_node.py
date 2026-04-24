# -*- coding: utf-8 -*-
"""
自动报表生成节点
生成效期报表、库存报表、合规台账
"""

import os
import traceback
from typing import Dict, Any, List, Optional
from datetime import datetime
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context

from graphs.state import (
    ReportGenerationInput,
    ReportGenerationOutput
)


def report_generation_node(state: ReportGenerationInput, config: RunnableConfig, runtime: Runtime[Context]) -> ReportGenerationOutput:
    """
    title: 自动报表生成
    desc: 生成效期报表、库存报表、合规台账，支持Excel/PDF导出
    """
    ctx = runtime.context
    
    print(f"[报表生成] 开始生成 {state.report_type} 报表...")
    
    try:
        start_time = datetime.now()
        reports = {}
        
        # 生成效期报表
        if state.report_type in ['expiry', 'all']:
            expiry_report_url = generate_expiry_report(
                state.expiry_data,
                state.alerts,
                state.export_format
            )
            reports['expiry'] = {'url': expiry_report_url}
        
        # 生成库存报表
        if state.report_type in ['inventory', 'all']:
            inventory_report_url = generate_inventory_report(
                state.quantity_stats,
                state.alerts,
                state.export_format
            )
            reports['inventory'] = {'url': inventory_report_url}
        
        # 生成合规台账
        if state.report_type in ['compliance', 'all']:
            compliance_report_url = generate_compliance_report(
                state.expiry_data,
                state.quantity_stats,
                state.alerts,
                state.export_format
            )
            reports['compliance'] = {'url': compliance_report_url}
        
        # 生成合并报表
        if state.report_type == 'all':
            combined_report_url = generate_combined_report(
                reports,
                state.export_format
            )
            reports['combined'] = {'url': combined_report_url}
        
        generation_time = (datetime.now() - start_time).total_seconds()
        
        print(f"[报表生成] 完成，耗时 {generation_time:.2f} 秒")
        
        return ReportGenerationOutput(
            reports=reports,
            expiry_report_url=reports.get('expiry', {}).get('url'),
            inventory_report_url=reports.get('inventory', {}).get('url'),
            compliance_report_url=reports.get('compliance', {}).get('url'),
            combined_report_url=reports.get('combined', {}).get('url'),
            generation_time=generation_time
        )
        
    except Exception as e:
        error_msg = f"报表生成发生错误: {str(e)}\n{traceback.format_exc()}"
        print(f"[报表生成] 错误: {error_msg}")
        
        return ReportGenerationOutput(
            reports={},
            expiry_report_url=None,
            inventory_report_url=None,
            compliance_report_url=None,
            combined_report_url=None,
            generation_time=0.0
        )


def generate_expiry_report(expiry_data: List[Dict[str, Any]], alerts: List[Dict[str, Any]], export_format: str) -> Optional[str]:
    """生成效期报表"""
    try:
        import pandas as pd
        import tempfile
        from coze_coding_dev_sdk.s3 import S3SyncStorage
        
        # 准备数据
        report_data = []
        for item in expiry_data:
            expiry_info = item.get('expiry_info', {})
            ocr_result = item.get('ocr_result', {})
            
            report_data.append({
                '区域编号': item.get('region_index', ''),
                '识别文本': ocr_result.get('text', ''),
                '生产日期': expiry_info.get('production_date', ''),
                '有效期': expiry_info.get('expiry_date', ''),
                '状态': translate_status(expiry_info.get('status', 'unknown')),
                '剩余天数': expiry_info.get('days_to_expiry', ''),
                '识别置信度': f"{ocr_result.get('confidence', 0.0):.2f}"
            })
        
        # 添加告警信息
        expiry_alerts = [a for a in alerts if a.get('type') == 'expiry']
        if expiry_alerts:
            for alert in expiry_alerts:
                report_data.append({
                    '区域编号': '告警信息',
                    '识别文本': alert.get('message', ''),
                    '生产日期': '',
                    '有效期': '',
                    '状态': alert.get('level', ''),
                    '剩余天数': '',
                    '识别置信度': ''
                })
        
        # 创建DataFrame
        df = pd.DataFrame(report_data)
        
        # 保存文件
        if export_format == 'excel':
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.xlsx', delete=False) as f:
                temp_path = f.name
                df.to_excel(f, index=False, engine='openpyxl')
        else:
            # PDF格式
            return generate_pdf_report(df, 'expiry')
        
        # 上传到对象存储
        storage = S3SyncStorage()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        object_name = f"reports/expiry_report_{timestamp}.xlsx"
        url = storage.upload_file(temp_path, object_name)
        
        # 删除临时文件
        try:
            os.unlink(temp_path)
        except:
            pass
        
        print(f"[报表生成] 效期报表已生成: {url}")
        return url
        
    except Exception as e:
        print(f"生成效期报表失败: {str(e)}")
        return None


def generate_inventory_report(quantity_stats: Dict[str, Any], alerts: List[Dict[str, Any]], export_format: str) -> Optional[str]:
    """生成库存报表"""
    try:
        import pandas as pd
        import tempfile
        from coze_coding_dev_sdk.s3 import S3SyncStorage
        
        # 准备数据
        report_data = [
            {
                '项目': '总商品数',
                '数值': quantity_stats.get('total_items', 0),
                '单位': '件',
                '备注': '货架/托盘总商品数'
            },
            {
                '项目': '分类数量',
                '数值': len(quantity_stats.get('category_breakdown', [])),
                '单位': '类',
                '备注': '商品分类数'
            }
        ]
        
        # 添加分类详情
        for category in quantity_stats.get('category_breakdown', []):
            report_data.append({
                '项目': f"分类: {category.get('name', 'unknown')}",
                '数值': category.get('count', 0),
                '单位': '件',
                '备注': category.get('description', '')
            })
        
        # 添加库存告警
        inventory_alerts = [a for a in alerts if a.get('type') == 'inventory']
        if inventory_alerts:
            for alert in inventory_alerts:
                report_data.append({
                    '项目': f"告警: {alert.get('title', '')}",
                    '数值': alert.get('current_stock', ''),
                    '单位': '',
                    '备注': alert.get('message', '')
                })
        
        # 创建DataFrame
        df = pd.DataFrame(report_data)
        
        # 保存文件
        if export_format == 'excel':
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.xlsx', delete=False) as f:
                temp_path = f.name
                df.to_excel(f, index=False, engine='openpyxl')
        else:
            return generate_pdf_report(df, 'inventory')
        
        # 上传到对象存储
        storage = S3SyncStorage()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        object_name = f"reports/inventory_report_{timestamp}.xlsx"
        url = storage.upload_file(temp_path, object_name)
        
        # 删除临时文件
        try:
            os.unlink(temp_path)
        except:
            pass
        
        print(f"[报表生成] 库存报表已生成: {url}")
        return url
        
    except Exception as e:
        print(f"生成库存报表失败: {str(e)}")
        return None


def generate_compliance_report(
    expiry_data: List[Dict[str, Any]],
    quantity_stats: Dict[str, Any],
    alerts: List[Dict[str, Any]],
    export_format: str
) -> Optional[str]:
    """生成合规台账"""
    try:
        import pandas as pd
        import tempfile
        from coze_coding_dev_sdk.s3 import S3SyncStorage
        
        # 准备数据
        report_data = []
        
        # 基本信息行
        report_data.append({
            '项目': '检测时间',
            '内容': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            '状态': '已完成',
            '备注': ''
        })
        report_data.append({
            '项目': '总商品数',
            '内容': str(len(expiry_data)) + ' 件',
            '状态': '已统计',
            '备注': ''
        })
        
        # 统计合规信息
        expired_count = sum(1 for item in expiry_data if item.get('expiry_info', {}).get('status') == 'expired')
        valid_count = sum(1 for item in expiry_data if item.get('expiry_info', {}).get('status') == 'valid')
        near_expiry_count = sum(1 for item in expiry_data if item.get('expiry_info', {}).get('status') == 'near_expiry')
        
        report_data.append({
            '项目': '过期商品',
            '内容': str(expired_count) + ' 件',
            '状态': '需处理' if expired_count > 0 else '正常',
            '备注': f'过期率 {expired_count/len(expiry_data)*100:.1f}%' if expiry_data else 'N/A'
        })
        
        report_data.append({
            '项目': '有效期内',
            '内容': str(valid_count) + ' 件',
            '状态': '正常',
            '备注': ''
        })
        
        report_data.append({
            '项目': '临期商品',
            '内容': str(near_expiry_count) + ' 件',
            '状态': '需关注' if near_expiry_count > 0 else '正常',
            '备注': '30天内过期'
        })
        
        # 添加合规告警
        compliance_alerts = [a for a in alerts if a.get('type') == 'compliance']
        if compliance_alerts:
            for alert in compliance_alerts:
                report_data.append({
                    '项目': f'合规告警: {alert.get("title", "")}',
                    '内容': alert.get('message', ''),
                    '状态': alert.get('level', ''),
                    '备注': f'过期率 {alert.get("expiry_rate", 0)*100:.1f}%'
                })
        
        # 创建DataFrame
        df = pd.DataFrame(report_data)
        
        # 保存文件
        if export_format == 'excel':
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.xlsx', delete=False) as f:
                temp_path = f.name
                df.to_excel(f, index=False, engine='openpyxl')
        else:
            return generate_pdf_report(df, 'compliance')
        
        # 上传到对象存储
        storage = S3SyncStorage()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        object_name = f"reports/compliance_report_{timestamp}.xlsx"
        url = storage.upload_file(temp_path, object_name)
        
        # 删除临时文件
        try:
            os.unlink(temp_path)
        except:
            pass
        
        print(f"[报表生成] 合规台账已生成: {url}")
        return url
        
    except Exception as e:
        print(f"生成合规台账失败: {str(e)}")
        return None


def generate_combined_report(reports: Dict[str, Any], export_format: str) -> Optional[str]:
    """生成合并报表"""
    try:
        import pandas as pd
        import tempfile
        from coze_coding_dev_sdk.s3 import S3SyncStorage
        
        # 创建合并报表
        combined_data = []
        
        # 添加各报表URL
        for report_type, report_info in reports.items():
            combined_data.append({
                '报表类型': report_type.upper(),
                '报表名称': get_report_name(report_type),
                '下载链接': report_info.get('url', ''),
                '生成时间': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
        
        # 创建DataFrame
        df = pd.DataFrame(combined_data)
        
        # 保存文件
        if export_format == 'excel':
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.xlsx', delete=False) as f:
                temp_path = f.name
                df.to_excel(f, index=False, engine='openpyxl')
        else:
            return generate_pdf_report(df, 'combined')
        
        # 上传到对象存储
        storage = S3SyncStorage()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        object_name = f"reports/combined_report_{timestamp}.xlsx"
        url = storage.upload_file(temp_path, object_name)
        
        # 删除临时文件
        try:
            os.unlink(temp_path)
        except:
            pass
        
        print(f"[报表生成] 合并报表已生成: {url}")
        return url
        
    except Exception as e:
        print(f"生成合并报表失败: {str(e)}")
        return None


def generate_pdf_report(df, report_type: str) -> Optional[str]:
    """生成PDF报表"""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.units import cm
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, PageBreak
        from reportlab.lib.styles import getSampleStyleSheet
        import tempfile
        from coze_coding_dev_sdk.s3 import S3SyncStorage
        
        # 保存到临时文件
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pdf', delete=False) as f:
            temp_path = f.name
            
            doc = SimpleDocTemplate(f, pagesize=A4)
            styles = getSampleStyleSheet()
            story = []
            
            # 标题
            title = Paragraph(f"{get_report_name(report_type)}", styles['Heading1'])
            story.append(title)
            story.append(Paragraph(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
            story.append(Paragraph("<br/><br/>", styles['Normal']))
            
            # 表格数据
            table_data = []
            for col in df.columns:
                table_data.append([col])
            for _, row in df.iterrows():
                for idx, col in enumerate(df.columns):
                    table_data[idx].append(str(row[col]))
            
            # 创建表格
            table = Table(table_data, colWidths=[3*cm] * len(df.columns))
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            
            story.append(table)
            
            # 构建PDF
            doc.build(story)
        
        # 上传到对象存储
        storage = S3SyncStorage()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        object_name = f"reports/{report_type}_report_{timestamp}.pdf"
        url = storage.upload_file(temp_path, object_name)
        
        # 删除临时文件
        try:
            os.unlink(temp_path)
        except:
            pass
        
        return url
        
    except Exception as e:
        print(f"生成PDF报表失败: {str(e)}")
        return None


def translate_status(status: str) -> str:
    """翻译状态"""
    status_map = {
        'valid': '有效',
        'near_expiry': '临期',
        'expired': '过期',
        'unknown': '未知'
    }
    return status_map.get(status, status)


def get_report_name(report_type: str) -> str:
    """获取报表名称"""
    name_map = {
        'expiry': '效期报表',
        'inventory': '库存报表',
        'compliance': '合规台账',
        'combined': '合并报表'
    }
    return name_map.get(report_type, report_type)
