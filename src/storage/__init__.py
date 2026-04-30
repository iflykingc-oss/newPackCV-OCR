# -*- coding: utf-8 -*-
"""
存储层封装
统一管理数据库、对象存储、缓存
"""

from src.storage.oss import OSSStorage, get_oss_storage
from src.storage.db import Database, get_database
from src.storage.cache import CacheManager, get_cache

__all__ = [
    'OSSStorage',
    'get_oss_storage',
    'Database',
    'get_database',
    'CacheManager',
    'get_cache'
]
