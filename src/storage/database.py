# -*- coding: utf-8 -*-
"""
数据库初始化和连接管理
"""

import os
from typing import Optional
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager

from .models import Base


# 数据库连接配置
DATABASE_URL = os.getenv(
    'DATABASE_URL',
    'postgresql://user:password@localhost:5432/ocr_system'
)


class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self, database_url: str = None):
        self.database_url = database_url or DATABASE_URL
        self.engine = None
        self.SessionLocal = None
        self._initialized = False
    
    def initialize(self):
        """初始化数据库连接"""
        if self._initialized:
            return
        
        # 创建引擎
        self.engine = create_engine(
            self.database_url,
            poolclass=QueuePool,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=False  # 设置为True可查看SQL日志
        )
        
        # 创建Session工厂
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        
        self._initialized = True
    
    def create_tables(self):
        """创建所有表"""
        if not self._initialized:
            self.initialize()
        
        Base.metadata.create_all(bind=self.engine)
        print("数据库表创建成功")
    
    def drop_tables(self):
        """删除所有表（谨慎使用）"""
        if not self._initialized:
            self.initialize()
        
        Base.metadata.drop_all(bind=self.engine)
        print("数据库表已删除")
    
    def execute_schema(self, schema_file: str):
        """执行SQL Schema文件"""
        if not self._initialized:
            self.initialize()
        
        with open(schema_file, 'r', encoding='utf-8') as f:
            schema_sql = f.read()
        
        with self.engine.connect() as conn:
            # 执行SQL
            conn.execute(text(schema_sql))
            conn.commit()
        
        print(f"Schema文件 {schema_file} 执行成功")
    
    @contextmanager
    def get_session(self) -> Session:
        """获取数据库会话（上下文管理器）"""
        if not self._initialized:
            self.initialize()
        
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def get_session_sync(self) -> Session:
        """获取数据库会话（同步，需手动关闭）"""
        if not self._initialized:
            self.initialize()
        return self.SessionLocal()
    
    def test_connection(self) -> bool:
        """测试数据库连接"""
        try:
            if not self._initialized:
                self.initialize()
            
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            print("数据库连接测试成功")
            return True
        except Exception as e:
            print(f"数据库连接测试失败: {str(e)}")
            return False
    
    def close(self):
        """关闭数据库连接"""
        if self.engine:
            self.engine.dispose()
            self._initialized = False


# 全局数据库管理器实例
db_manager = DatabaseManager()


# 初始化数据库
def init_database():
    """初始化数据库"""
    db_manager.initialize()
    db_manager.create_tables()


def get_db():
    """获取数据库会话（用于FastAPI等框架）"""
    session = db_manager.get_session_sync()
    try:
        yield session
    finally:
        session.close()


if __name__ == "__main__":
    # 测试数据库连接和初始化
    print("开始初始化数据库...")
    
    db_manager = DatabaseManager()
    
    # 测试连接
    if db_manager.test_connection():
        print("✓ 数据库连接成功")
    else:
        print("✗ 数据库连接失败")
        exit(1)
    
    # 创建表
    print("正在创建表...")
    db_manager.create_tables()
    print("✓ 表创建完成")
    
    # 关闭连接
    db_manager.close()
