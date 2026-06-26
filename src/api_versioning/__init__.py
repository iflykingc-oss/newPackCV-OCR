"""API 版本管理模块"""
from api_versioning.manager import APIVersionManager, version_manager, DeprecationLevel, init_default_endpoints

__all__ = ["APIVersionManager", "version_manager", "DeprecationLevel", "init_default_endpoints"]
