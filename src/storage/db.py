# -*- coding: utf-8 -*-
"""
数据库封装
支持PostgreSQL/MySQL/SQLite
"""

import os
import json
import logging
from typing import Optional, List, Dict, Any, Type
from datetime import datetime
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class Database:
    """数据库封装类"""

    def __init__(
        self,
        host: str = None,
        port: int = 5432,
        database: str = None,
        user: str = None,
        password: str = None
    ):
        """
        初始化数据库连接

        Args:
            host: 数据库主机
            port: 端口
            database: 数据库名
            user: 用户名
            password: 密码
        """
        self.host = host or os.getenv("DB_HOST", "localhost")
        self.port = port or int(os.getenv("DB_PORT", "5432"))
        self.database = database or os.getenv("DB_NAME", "packcv")
        self.user = user or os.getenv("DB_USER", "postgres")
        self.password = password or os.getenv("DB_PASSWORD", "")

        self._engine = None
        self._session = None
        self._init_engine()

    def _init_engine(self):
        """初始化数据库引擎"""
        try:
            from sqlalchemy import create_engine
            from sqlalchemy.orm import sessionmaker

            url = f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
            self._engine = create_engine(url, pool_pre_ping=True, pool_size=10)
            self._session_factory = sessionmaker(bind=self._engine)
            logger.info(f"数据库连接成功: {self.host}:{self.port}/{self.database}")
        except ImportError:
            logger.warning("SQLAlchemy未安装，使用模拟模式")
            self._engine = None

    @contextmanager
    def get_session(self):
        """获取数据库会话"""
        if not self._engine:
            yield None
            return

        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"数据库操作失败: {e}")
            raise
        finally:
            session.close()

    def execute(self, sql: str, params: tuple = None) -> Any:
        """执行SQL"""
        if not self._engine:
            logger.warning("数据库未连接，跳过执行")
            return None

        with self._engine.connect() as conn:
            result = conn.execute(sql, params or {})
            return result

    def execute_query(self, sql: str, params: tuple = None) -> List[Dict]:
        """执行查询"""
        if not self._engine:
            return []

        with self._engine.connect() as conn:
            result = conn.execute(sql, params or {})
            columns = result.keys()
            return [dict(zip(columns, row)) for row in result]


# 全局单例
_db_instance: Optional[Database] = None


def get_database() -> Database:
    """获取数据库单例"""
    global _db_instance
    if _db_instance is None:
        _db_instance = Database()
    return _db_instance


# ============ 数据模型定义 ============

class OCRResultModel:
    """OCR识别结果模型"""

    def __init__(
        self,
        id: str = None,
        image_url: str = None,
        raw_text: str = None,
        corrected_text: str = None,
        structured_data: Dict[str, Any] = None,
        confidence: float = 0.0,
        engine: str = None,
        status: str = "pending",
        error_message: str = None,
        created_at: datetime = None,
        updated_at: datetime = None
    ):
        self.id = id
        self.image_url = image_url
        self.raw_text = raw_text
        self.corrected_text = corrected_text
        self.structured_data = structured_data or {}
        self.confidence = confidence
        self.engine = engine
        self.status = status
        self.error_message = error_message
        self.created_at = created_at or datetime.now()
        self.updated_at = updated_at or datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "image_url": self.image_url,
            "raw_text": self.raw_text,
            "corrected_text": self.corrected_text,
            "structured_data": self.structured_data,
            "confidence": self.confidence,
            "engine": self.engine,
            "status": self.status,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OCRResultModel':
        data = data.copy()
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if 'updated_at' in data and isinstance(data['updated_at'], str):
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        return cls(**data)


class AlertRecordModel:
    """告警记录模型"""

    def __init__(
        self,
        id: str = None,
        ocr_result_id: str = None,
        alert_type: str = None,  # expired/approaching/low_stock
        alert_level: str = "warning",  # critical/warning/info
        message: str = None,
        product_name: str = None,
        product_date: str = None,
        shelf_life: str = None,
        status: str = "pending",  # pending/acknowledged/resolved
        acknowledged_by: str = None,
        acknowledged_at: datetime = None,
        resolved_at: datetime = None,
        created_at: datetime = None
    ):
        self.id = id
        self.ocr_result_id = ocr_result_id
        self.alert_type = alert_type
        self.alert_level = alert_level
        self.message = message
        self.product_name = product_name
        self.product_date = product_date
        self.shelf_life = shelf_life
        self.status = status
        self.acknowledged_by = acknowledged_by
        self.acknowledged_at = acknowledged_at
        self.resolved_at = resolved_at
        self.created_at = created_at or datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "ocr_result_id": self.ocr_result_id,
            "alert_type": self.alert_type,
            "alert_level": self.alert_level,
            "message": self.message,
            "product_name": self.product_name,
            "product_date": self.product_date,
            "shelf_life": self.shelf_life,
            "status": self.status,
            "acknowledged_by": self.acknowledged_by,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class ReportModel:
    """报表模型"""

    def __init__(
        self,
        id: str = None,
        report_type: str = None,  # expiry/stock/compliance
        title: str = None,
        content: Dict[str, Any] = None,
        file_url: str = None,
        period_start: datetime = None,
        period_end: datetime = None,
        created_at: datetime = None
    ):
        self.id = id
        self.report_type = report_type
        self.title = title
        self.content = content or {}
        self.file_url = file_url
        self.period_start = period_start
        self.period_end = period_end
        self.created_at = created_at or datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "report_type": self.report_type,
            "title": self.title,
            "content": self.content,
            "file_url": self.file_url,
            "period_start": self.period_start.isoformat() if self.period_start else None,
            "period_end": self.period_end.isoformat() if self.period_end else None,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }
