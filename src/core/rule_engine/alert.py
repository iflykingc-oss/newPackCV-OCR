# -*- coding: utf-8 -*-
"""
告警管理模块
提供告警生成、级别管理、触达能力
"""

import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime


logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """告警级别"""
    CRITICAL = "critical"   # 紧急：商品已过期
    WARNING = "warning"      # 警告：商品临期
    INFO = "info"            # 提示：信息异常


@dataclass
class Alert:
    """告警对象"""
    level: AlertLevel
    title: str
    message: str
    field_name: str
    field_value: Any
    threshold: Any = None
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'level': self.level.value,
            'title': self.title,
            'message': self.message,
            'field_name': self.field_name,
            'field_value': str(self.field_value),
            'threshold': str(self.threshold) if self.threshold else None,
            'created_at': self.created_at.isoformat(),
            'acknowledged': self.acknowledged,
            'acknowledged_by': self.acknowledged_by,
            'acknowledged_at': self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            'metadata': self.metadata
        }


class AlertManager:
    """
    告警管理器
    支持：
    - 多级别告警生成
    - 告警过滤与去重
    - 告警确认与归档
    - 统计与分析
    """

    # 临期阈值（天）
    DEFAULT_EXPIRY_WARNING_DAYS = 30
    DEFAULT_EXPIRY_CRITICAL_DAYS = 0  # 0表示已过期

    def __init__(self):
        self.alerts: List[Alert] = []
        self.acknowledged_alerts: List[Alert] = []
        self._alert_count_by_level: Dict[AlertLevel, int] = {
            AlertLevel.CRITICAL: 0,
            AlertLevel.WARNING: 0,
            AlertLevel.INFO: 0
        }

    def add_alert(self, alert: Alert):
        """添加告警"""
        # 检查是否重复
        if self._is_duplicate(alert):
            logger.debug(f"[告警] 重复告警已忽略: {alert.title}")
            return

        self.alerts.append(alert)
        self._alert_count_by_level[alert.level] += 1
        logger.info(f"[告警] 新增告警 [{alert.level.value}]: {alert.title}")

    def _is_duplicate(self, alert: Alert) -> bool:
        """检查是否重复告警"""
        for existing in self.alerts:
            if (existing.field_name == alert.field_name and
                existing.field_value == alert.field_value and
                not existing.acknowledged):
                return True
        return False

    def check_expiry_alert(
        self,
        expiry_date: datetime,
        field_value: str,
        warning_days: int = None,
        critical_days: int = None
    ) -> Optional[Alert]:
        """
        检查效期告警
        Returns: Alert对象（如果有告警）
        """
        warning_days = warning_days or self.DEFAULT_EXPIRY_WARNING_DAYS
        critical_days = critical_days or self.DEFAULT_EXPIRY_CRITICAL_DAYS

        now = datetime.now()
        days_until_expiry = (expiry_date - now).days

        # 检查是否已过期
        if days_until_expiry <= critical_days:
            alert = Alert(
                level=AlertLevel.CRITICAL,
                title="商品已过期",
                message=f"商品效期已过 {-days_until_expiry} 天，请立即处理",
                field_name="expiry_date",
                field_value=field_value,
                threshold=f"{critical_days}天",
                metadata={'days_until_expiry': days_until_expiry}
            )
            self.add_alert(alert)
            return alert

        # 检查是否临期
        if days_until_expiry <= warning_days:
            alert = Alert(
                level=AlertLevel.WARNING,
                title="商品临期",
                message=f"商品效期还剩 {days_until_expiry} 天，请注意处理",
                field_name="expiry_date",
                field_value=field_value,
                threshold=f"{warning_days}天",
                metadata={'days_until_expiry': days_until_expiry}
            )
            self.add_alert(alert)
            return alert

        return None

    def check_confidence_alert(
        self,
        field_name: str,
        value: Any,
        confidence: float,
        threshold: float = 0.85
    ) -> Optional[Alert]:
        """
        检查置信度告警
        """
        if confidence < threshold:
            alert = Alert(
                level=AlertLevel.INFO,
                title=f"识别置信度低",
                message=f"字段 {field_name} 置信度 {confidence:.2%} 低于阈值 {threshold:.2%}",
                field_name=field_name,
                field_value=value,
                threshold=threshold,
                metadata={'confidence': confidence}
            )
            self.add_alert(alert)
            return alert

        return None

    def acknowledge_alert(
        self,
        alert_index: int,
        acknowledged_by: str = "system"
    ) -> bool:
        """确认告警"""
        if 0 <= alert_index < len(self.alerts):
            alert = self.alerts[alert_index]
            alert.acknowledged = True
            alert.acknowledged_by = acknowledged_by
            alert.acknowledged_at = datetime.now()

            self.acknowledged_alerts.append(alert)
            self.alerts.remove(alert)

            logger.info(f"[告警] 告警已确认: {alert.title} by {acknowledged_by}")
            return True

        return False

    def get_active_alerts(self) -> List[Alert]:
        """获取未确认的告警"""
        return [a for a in self.alerts if not a.acknowledged]

    def get_alerts_by_level(self, level: AlertLevel) -> List[Alert]:
        """按级别获取告警"""
        return [a for a in self.alerts if a.level == level and not a.acknowledged]

    def get_statistics(self) -> Dict[str, Any]:
        """获取告警统计"""
        return {
            'total': len(self.alerts) + len(self.acknowledged_alerts),
            'active': len(self.alerts),
            'acknowledged': len(self.acknowledged_alerts),
            'by_level': {
                level.value: len(self.get_alerts_by_level(level))
                for level in AlertLevel
            }
        }

    def clear_acknowledged(self, days: int = 30):
        """清理超过指定天数的已确认告警"""
        now = datetime.now()
        cutoff = now - timedelta(days=days)

        original_count = len(self.acknowledged_alerts)
        self.acknowledged_alerts = [
            a for a in self.acknowledged_alerts
            if a.acknowledged_at and a.acknowledged_at > cutoff
        ]

        removed = original_count - len(self.acknowledged_alerts)
        if removed > 0:
            logger.info(f"[告警] 清理了 {removed} 条已确认告警")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'statistics': self.get_statistics(),
            'active_alerts': [a.to_dict() for a in self.alerts],
            'acknowledged_alerts': [a.to_dict() for a in self.acknowledged_alerts[-100:]]  # 最近100条
        }


def create_alert_manager() -> AlertManager:
    """告警管理器工厂函数"""
    return AlertManager()
