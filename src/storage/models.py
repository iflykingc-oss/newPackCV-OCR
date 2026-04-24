# -*- coding: utf-8 -*-
"""
数据库ORM模型定义
使用SQLAlchemy定义所有数据库表模型
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy import (
    Column, Integer, BigInteger, String, Text, Boolean, 
    DateTime, Numeric, JSON, ForeignKey, Index, CheckConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from pydantic import BaseModel

Base = declarative_base()


# ========================================
# 1. 用户模型
# ========================================

class User(Base):
    """用户表"""
    __tablename__ = 'users'
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False, index=True)
    phone = Column(String(20), index=True)
    password_hash = Column(String(255), nullable=False)
    nickname = Column(String(100))
    avatar_url = Column(String(500))
    status = Column(String(20), default='active', index=True)  # active, inactive, banned
    account_type = Column(String(20), default='personal')  # personal, enterprise
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login_at = Column(DateTime)
    
    # 关系
    roles = relationship("Role", secondary="user_roles", back_populates="users")
    teams_owned = relationship("Team", foreign_keys="Team.owner_id")
    team_memberships = relationship("TeamMember", back_populates="user")
    ocr_records = relationship("OCRRecord", back_populates="user")
    configs = relationship("UserConfig", back_populates="user")
    platform_integrations = relationship("PlatformIntegration", back_populates="user")
    batch_tasks = relationship("BatchTask", back_populates="user")
    model_configs = relationship("ModelConfig", back_populates="user")
    ocr_api_configs = relationship("OCRApiConfig", back_populates="user")
    
    __table_args__ = (
        CheckConstraint("status IN ('active', 'inactive', 'banned')", name='valid_status'),
        CheckConstraint("account_type IN ('personal', 'enterprise')", name='valid_account_type'),
    )


# ========================================
# 2. 角色模型
# ========================================

class Role(Base):
    """角色表"""
    __tablename__ = 'roles'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(50), unique=True, nullable=False)
    description = Column(String(200))
    permissions = Column(JSON)  # 权限列表
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # 关系
    users = relationship("User", secondary="user_roles", back_populates="roles")


# ========================================
# 3. 用户角色关联表
# ========================================

from sqlalchemy import Table

user_roles = Table(
    'user_roles',
    Base.metadata,
    Column('user_id', BigInteger, ForeignKey('users.id', ondelete='CASCADE'), primary_key=True),
    Column('role_id', Integer, ForeignKey('roles.id', ondelete='CASCADE'), primary_key=True),
    Column('assigned_at', DateTime, default=datetime.utcnow)
)


# ========================================
# 4. 企业团队模型
# ========================================

class Team(Base):
    """团队表"""
    __tablename__ = 'teams'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    description = Column(String(500))
    owner_id = Column(BigInteger, ForeignKey('users.id'))
    member_limit = Column(Integer, default=100)
    status = Column(String(20), default='active')
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # 关系
    owner = relationship("User", foreign_keys=[owner_id])
    members = relationship("TeamMember", back_populates="team")
    ocr_records = relationship("OCRRecord", back_populates="team")
    platform_integrations = relationship("PlatformIntegration", back_populates="team")
    batch_tasks = relationship("BatchTask", back_populates="team")
    model_configs = relationship("ModelConfig", back_populates="team")
    ocr_api_configs = relationship("OCRApiConfig", back_populates="team")


# ========================================
# 5. 团队成员模型
# ========================================

class TeamMember(Base):
    """团队成员表"""
    __tablename__ = 'team_members'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    team_id = Column(Integer, ForeignKey('teams.id', ondelete='CASCADE'), nullable=False)
    user_id = Column(BigInteger, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    role = Column(String(20), default='member')  # owner, admin, member
    joined_at = Column(DateTime, default=datetime.utcnow)
    
    # 关系
    team = relationship("Team", back_populates="members")
    user = relationship("User", back_populates="team_memberships")
    
    __table_args__ = (
        Index('idx_team_membership', 'team_id', 'user_id', unique=True),
    )


# ========================================
# 6. OCR识别记录模型
# ========================================

class OCRRecord(Base):
    """OCR识别记录表"""
    __tablename__ = 'ocr_records'
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(BigInteger, ForeignKey('users.id', ondelete='SET NULL'))
    team_id = Column(Integer, ForeignKey('teams.id', ondelete='SET NULL'))
    
    # 输入信息
    image_url = Column(String(500), nullable=False)
    image_file_name = Column(String(200))
    image_size = Column(Integer)  # bytes
    image_width = Column(Integer)
    image_height = Column(Integer)
    
    # OCR引擎配置
    ocr_engine_type = Column(String(20), nullable=False)  # builtin, api
    ocr_engine_name = Column(String(50))
    ocr_api_config_id = Column(Integer)
    
    # OCR结果
    ocr_text = Column(Text)
    ocr_confidence = Column(Numeric(5, 2))
    ocr_regions = Column(JSON)
    processing_time = Column(Numeric(10, 2))  # seconds
    
    # 模型调用配置
    model_type = Column(String(20))  # extract, correct, qa
    model_name = Column(String(50))
    
    # 模型结果
    structured_data = Column(JSON)
    corrected_text = Column(Text)
    qa_answer = Column(Text)
    
    # 输出配置
    export_format = Column(String(10))  # json, excel, pdf
    export_file_url = Column(String(500))
    
    # 平台推送
    platform = Column(String(20))  # wechat, feishu, none
    push_status = Column(String(20))  # pending, success, failed
    push_result = Column(JSON)
    
    # 状态和元数据
    status = Column(String(20), default='success', index=True)  # success, failed, processing
    error_message = Column(Text)
    metadata = Column(JSON)  # 额外信息
    
    # 时间戳
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # 关系
    user = relationship("User", back_populates="ocr_records")
    team = relationship("Team", back_populates="ocr_records")
    
    __table_args__ = (
        Index('idx_ocr_records_user_id', 'user_id'),
        Index('idx_ocr_records_team_id', 'team_id'),
    )


# ========================================
# 7. 用户配置模型
# ========================================

class UserConfig(Base):
    """用户配置表"""
    __tablename__ = 'user_configs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(BigInteger, ForeignKey('users.id', ondelete='CASCADE'))
    config_key = Column(String(100), nullable=False)
    config_value = Column(JSON, nullable=False)
    description = Column(String(200))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # 关系
    user = relationship("User", back_populates="configs")
    
    __table_args__ = (
        Index('idx_user_configs_key', 'user_id', 'config_key', unique=True),
    )


# ========================================
# 8. 多平台集成配置模型
# ========================================

class PlatformIntegration(Base):
    """多平台集成配置表"""
    __tablename__ = 'platform_integrations'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(BigInteger, ForeignKey('users.id', ondelete='CASCADE'))
    team_id = Column(Integer, ForeignKey('teams.id', ondelete='CASCADE'))
    
    platform = Column(String(20), nullable=False, index=True)  # wechat, feishu
    integration_type = Column(String(50), nullable=False)  # webhook, api_key, oauth
    
    # 集成配置（加密存储）
    credentials = Column(JSON, nullable=False)  # webhook_url, api_key, app_id等
    is_enabled = Column(Boolean, default=True)
    
    # 元数据
    metadata = Column(JSON)
    last_used_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # 关系
    user = relationship("User", back_populates="platform_integrations")
    team = relationship("Team", back_populates="platform_integrations")
    
    __table_args__ = (
        Index('idx_platform_integrations_user_id', 'user_id'),
    )


# ========================================
# 9. 批量处理任务模型
# ========================================

class BatchTask(Base):
    """批量处理任务表"""
    __tablename__ = 'batch_tasks'
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(BigInteger, ForeignKey('users.id', ondelete='SET NULL'))
    team_id = Column(Integer, ForeignKey('teams.id', ondelete='SET NULL'))
    
    task_name = Column(String(200))
    status = Column(String(20), default='pending', index=True)  # pending, processing, completed, failed
    
    # 输入配置
    image_urls = Column(JSON, nullable=False)  # 图片URL列表
    total_count = Column(Integer, nullable=False)
    processed_count = Column(Integer, default=0)
    success_count = Column(Integer, default=0)
    failed_count = Column(Integer, default=0)
    
    # OCR配置
    ocr_engine_type = Column(String(20))
    ocr_api_config_id = Column(Integer)
    
    # 模型配置
    model_type = Column(String(20))
    model_name = Column(String(50))
    
    # 输出配置
    export_format = Column(String(10))
    merged_export_url = Column(String(500))
    
    # 错误信息
    error_message = Column(Text)
    errors = Column(JSON)  # 各图片的错误信息
    
    # 时间戳
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # 关系
    user = relationship("User", back_populates="batch_tasks")
    team = relationship("Team", back_populates="batch_tasks")
    results = relationship("BatchTaskResult", back_populates="batch_task", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_batch_tasks_user_id', 'user_id'),
    )


# ========================================
# 10. 批量处理结果详情模型
# ========================================

class BatchTaskResult(Base):
    """批量处理结果详情表"""
    __tablename__ = 'batch_task_results'
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    batch_task_id = Column(BigInteger, ForeignKey('batch_tasks.id', ondelete='CASCADE'))
    
    image_url = Column(String(500), nullable=False)
    image_index = Column(Integer, nullable=False)
    status = Column(String(20))  # success, failed
    
    # OCR结果
    ocr_text = Column(Text)
    ocr_confidence = Column(Numeric(5, 2))
    
    # 模型结果
    structured_data = Column(JSON)
    
    # 错误信息
    error_message = Column(Text)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # 关系
    batch_task = relationship("BatchTask", back_populates="results")
    
    __table_args__ = (
        Index('idx_batch_task_results_batch_task_id', 'batch_task_id'),
    )


# ========================================
# 11. 模型配置模型（用户自定义）
# ========================================

class ModelConfig(Base):
    """模型配置表（用户自定义）"""
    __tablename__ = 'model_configs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(BigInteger, ForeignKey('users.id', ondelete='CASCADE'))
    team_id = Column(Integer, ForeignKey('teams.id', ondelete='SET NULL'))
    
    name = Column(String(100), nullable=False)
    model_type = Column(String(20), nullable=False)  # extract, correct, qa
    model_name = Column(String(50), nullable=False)
    
    # 模型参数
    temperature = Column(Numeric(3, 2))
    max_tokens = Column(Integer)
    top_p = Column(Numeric(3, 2))
    
    # 提示词配置
    system_prompt = Column(Text)
    user_prompt_template = Column(Text)
    
    # 其他配置
    metadata = Column(JSON)
    is_default = Column(Boolean, default=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # 关系
    user = relationship("User", back_populates="model_configs")
    team = relationship("Team", back_populates="model_configs")
    
    __table_args__ = (
        Index('idx_model_configs_user_id', 'user_id'),
    )


# ========================================
# 12. OCR API配置模型（用户自定义）
# ========================================

class OCRApiConfig(Base):
    """OCR API配置表（用户自定义）"""
    __tablename__ = 'ocr_api_configs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(BigInteger, ForeignKey('users.id', ondelete='CASCADE'))
    team_id = Column(Integer, ForeignKey('teams.id', ondelete='SET NULL'))
    
    name = Column(String(100), nullable=False)
    api_url = Column(String(500), nullable=False)
    api_key = Column(String(500))
    
    # API配置
    headers = Column(JSON)
    parameters = Column(JSON)
    
    # 元数据
    description = Column(String(500))
    is_default = Column(Boolean, default=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # 关系
    user = relationship("User", back_populates="ocr_api_configs")
    team = relationship("Team", back_populates="ocr_api_configs")
    
    __table_args__ = (
        Index('idx_ocr_api_configs_user_id', 'user_id'),
    )
