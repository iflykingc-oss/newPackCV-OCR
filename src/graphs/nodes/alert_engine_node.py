# -*- coding: utf-8 -*-
"""
智能告警引擎节点
根据效期数据和库存状态生成智能告警
"""

import traceback
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context

from graphs.state import (
    AlertEngineInput,
    AlertEngineOutput
)


def alert_engine_node(state: AlertEngineInput, config: RunnableConfig, runtime: Runtime[Context]) -> AlertEngineOutput:
    """
    title: 智能告警引擎
    desc: 根据效期数据和库存状态生成智能告警（效期/库存/合规）
    """
    ctx = runtime.context
    
    print(f"[告警引擎] 开始分析告警规则...")
    
    try:
        alerts = []
        critical_count = 0
        warning_count = 0
        info_count = 0
        
        # 1. 效期告警
        expiry_alerts = analyse_expiry_alerts(
            state.expiry_data,
            state.near_expiry_days
        )
        alerts.extend(expiry_alerts)
        
        # 2. 库存告警
        inventory_alerts = analyse_inventory_alerts(
            state.quantity_stats,
            state.inventory_status,
            state.low_stock_threshold,
            state.alert_rules
        )
        alerts.extend(inventory_alerts)
        
        # 3. 合规告警
        compliance_alerts = analyse_compliance_alerts(
            state.expiry_data,
            state.quantity_stats
        )
        alerts.extend(compliance_alerts)
        
        # 统计告警级别
        for alert in alerts:
            level = alert.get('level', 'info')
            if level == 'critical':
                critical_count += 1
            elif level == 'warning':
                warning_count += 1
            else:
                info_count += 1
        
        # 生成告警摘要
        alert_summary = generate_alert_summary(alerts, state.expiry_data, state.quantity_stats)
        
        print(f"[告警引擎] 完成，生成 {len(alerts)} 条告警（严重: {critical_count}, 警告: {warning_count}, 信息: {info_count}）")
        
        return AlertEngineOutput(
            alerts=alerts,
            alert_summary=alert_summary,
            critical_count=critical_count,
            warning_count=warning_count,
            info_count=info_count
        )
        
    except Exception as e:
        error_msg = f"告警引擎发生错误: {str(e)}\n{traceback.format_exc()}"
        print(f"[告警引擎] 错误: {error_msg}")
        
        return AlertEngineOutput(
            alerts=[],
            alert_summary={},
            critical_count=0,
            warning_count=0,
            info_count=0
        )


def analyse_expiry_alerts(expiry_data: List[Dict[str, Any]], near_expiry_days: int) -> List[Dict[str, Any]]:
    """分析效期告警"""
    alerts = []
    
    if not expiry_data:
        return alerts
    
    expired_count = 0
    near_expiry_count = 0
    expired_products = []
    near_expiry_products = []
    
    for item in expiry_data:
        expiry_info = item.get('expiry_info', {})
        if not expiry_info:
            continue
        
        status = expiry_info.get('status', 'unknown')
        expiry_date = expiry_info.get('expiry_date')
        days_to_expiry = expiry_info.get('days_to_expiry')
        
        if status == 'expired':
            expired_count += 1
            expired_products.append({
                'region_index': item.get('region_index'),
                'expiry_date': expiry_date,
                'days_overdue': abs(days_to_expiry) if days_to_expiry else 0
            })
        elif status == 'near_expiry':
            near_expiry_count += 1
            near_expiry_products.append({
                'region_index': item.get('region_index'),
                'expiry_date': expiry_date,
                'days_to_expiry': days_to_expiry
            })
    
    # 过期告警（严重）
    if expired_count > 0:
        alerts.append({
            'id': f'expiry_critical_{datetime.now().timestamp()}',
            'type': 'expiry',
            'level': 'critical',
            'title': f'发现 {expired_count} 个过期商品',
            'message': f'检测到 {expired_count} 个商品已过期，需立即处理',
            'count': expired_count,
            'products': expired_products[:10],  # 最多显示10个
            'timestamp': datetime.now().isoformat()
        })
    
    # 临期告警（警告）
    if near_expiry_count > 0:
        alerts.append({
            'id': f'expiry_warning_{datetime.now().timestamp()}',
            'type': 'expiry',
            'level': 'warning',
            'title': f'发现 {near_expiry_count} 个临期商品',
            'message': f'检测到 {near_expiry_count} 个商品将在 {near_expiry_days} 天内过期',
            'count': near_expiry_count,
            'products': near_expiry_products[:10],
            'timestamp': datetime.now().isoformat()
        })
    
    return alerts


def analyse_inventory_alerts(
    quantity_stats: Dict[str, Any],
    inventory_status: Dict[str, Any],
    low_stock_threshold: int,
    alert_rules: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """分析库存告警"""
    alerts = []
    
    if not quantity_stats:
        return alerts
    
    total_items = quantity_stats.get('total_items', 0)
    
    # 低库存告警
    if total_items > 0 and total_items < low_stock_threshold:
        alerts.append({
            'id': f'inventory_low_{datetime.now().timestamp()}',
            'type': 'inventory',
            'level': 'warning',
            'title': f'库存不足预警',
            'message': f'当前库存 {total_items} 件，低于最低阈值 {low_stock_threshold} 件',
            'current_stock': total_items,
            'threshold': low_stock_threshold,
            'timestamp': datetime.now().isoformat()
        })
    
    # 压货告警（自定义规则）
    if alert_rules.get('overstock_enabled', False):
        overstock_threshold = alert_rules.get('overstock_threshold', 100)
        if total_items > overstock_threshold:
            alerts.append({
                'id': f'inventory_overstock_{datetime.now().timestamp()}',
                'type': 'inventory',
                'level': 'info',
                'title': f'压货风险提示',
                'message': f'当前库存 {total_items} 件，超过压货阈值 {overstock_threshold} 件',
                'current_stock': total_items,
                'threshold': overstock_threshold,
                'timestamp': datetime.now().isoformat()
            })
    
    # 缺货告警
    if inventory_status.get('out_of_stock', False):
        alerts.append({
            'id': f'inventory_out_of_stock_{datetime.now().timestamp()}',
            'type': 'inventory',
            'level': 'critical',
            'title': f'缺货告警',
            'message': f'检测到缺货情况，请及时补货',
            'out_of_stock_items': inventory_status.get('out_of_stock_items', []),
            'timestamp': datetime.now().isoformat()
        })
    
    return alerts


def analyse_compliance_alerts(
    expiry_data: List[Dict[str, Any]],
    quantity_stats: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """分析合规告警"""
    alerts = []
    
    if not expiry_data:
        return alerts
    
    total_products = len(expiry_data)
    expired_count = sum(1 for item in expiry_data if item.get('expiry_info', {}).get('status') == 'expired')
    
    # 过期率告警
    if total_products > 0:
        expiry_rate = expired_count / total_products
        if expiry_rate > 0.05:  # 超过5%
            alerts.append({
                'id': f'compliance_expiry_rate_{datetime.now().timestamp()}',
                'type': 'compliance',
                'level': 'critical' if expiry_rate > 0.1 else 'warning',
                'title': f'过期率超标告警',
                'message': f'过期率 {expiry_rate*100:.1f}%，超过合规阈值 5%',
                'expiry_rate': expiry_rate,
                'expired_count': expired_count,
                'total_count': total_products,
                'timestamp': datetime.now().isoformat()
            })
    
    return alerts


def generate_alert_summary(
    alerts: List[Dict[str, Any]],
    expiry_data: List[Dict[str, Any]],
    quantity_stats: Dict[str, Any]
) -> Dict[str, Any]:
    """生成告警摘要"""
    summary = {
        'total_alerts': len(alerts),
        'critical_alerts': sum(1 for a in alerts if a.get('level') == 'critical'),
        'warning_alerts': sum(1 for a in alerts if a.get('level') == 'warning'),
        'info_alerts': sum(1 for a in alerts if a.get('level') == 'info'),
        'alert_types': {},
        'recommendations': []
    }
    
    # 按类型统计
    for alert in alerts:
        alert_type = alert.get('type', 'unknown')
        summary['alert_types'][alert_type] = summary['alert_types'].get(alert_type, 0) + 1
    
    # 生成建议
    if summary['critical_alerts'] > 0:
        summary['recommendations'].append('立即处理严重告警，特别是过期商品')
    if summary['warning_alerts'] > 0:
        summary['recommendations'].append('关注警告告警，做好预防措施')
    
    if expiry_data:
        expired_count = sum(1 for item in expiry_data if item.get('expiry_info', {}).get('status') == 'expired')
        if expired_count > 0:
            summary['recommendations'].append(f'立即下架 {expired_count} 个过期商品')
    
    return summary
