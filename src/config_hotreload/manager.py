#!/usr/bin/env python3
"""配置热更新模块
功能:
- 无重启更新 LLM Provider 配置
- 无重启更新限流阈值
- 配置版本管理
- 配置变更通知
"""
import os
import json
import logging
import threading
import time
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, Field
import hashlib

logger = logging.getLogger(__name__)

# 配置文件路径
CONFIG_DIR = Path(os.getenv("COZE_WORKSPACE_PATH", "")) / "config"
DATA_DIR = Path(os.getenv("COZE_WORKSPACE_PATH", "")) / "data"

# 配置缓存
_config_cache: Dict[str, Any] = {}
_config_versions: Dict[str, str] = {}
_config_watchers: List[Callable] = []
_watch_thread: Optional[threading.Thread] = None
_running = False


class ConfigChange(BaseModel):
    """配置变更记录"""
    config_name: str = Field(..., description="配置名称")
    version: str = Field(..., description="版本号 (MD5)")
    changed_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    changes: Dict[str, Any] = Field(default_factory=dict, description="变更内容")
    changed_by: str = Field(default="system", description="变更来源")


class HotReloadManager:
    """配置热更新管理器"""
    
    @staticmethod
    def load_config(config_name: str) -> Dict[str, Any]:
        """加载配置
        
        Args:
            config_name: 配置名称 (如 llm_providers, rate_limits)
        
        Returns:
            配置字典
        """
        # 尝试多个路径
        paths = [
            DATA_DIR / f"{config_name}.json",
            CONFIG_DIR / f"{config_name}.json",
        ]
        
        for path in paths:
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    # 计算版本
                    version = hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]
                    _config_cache[config_name] = config
                    _config_versions[config_name] = version
                    logger.debug(f"Loaded config {config_name} v{version}")
                    return config
        
        logger.warning(f"Config {config_name} not found")
        return {}
    
    @staticmethod
    def get_config(config_name: str, reload: bool = False) -> Dict[str, Any]:
        """获取配置
        
        Args:
            config_name: 配置名称
            reload: 是否强制重新加载
        
        Returns:
            配置字典
        """
        if reload or config_name not in _config_cache:
            return HotReloadManager.load_config(config_name)
        return _config_cache.get(config_name, {})
    
    @staticmethod
    def update_config(
        config_name: str,
        updates: Dict[str, Any],
        changed_by: str = "api"
    ) -> ConfigChange:
        """更新配置
        
        Args:
            config_name: 配置名称
            updates: 更新内容
            changed_by: 变更来源
        
        Returns:
            变更记录
        """
        # 加载当前配置
        current = HotReloadManager.get_config(config_name)
        
        # 合并更新
        new_config = {**current, **updates}
        
        # 计算新版本
        new_version = hashlib.md5(json.dumps(new_config, sort_keys=True).encode()).hexdigest()[:8]
        old_version = _config_versions.get(config_name, "unknown")
        
        # 写入文件
        path = DATA_DIR / f"{config_name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(new_config, f, indent=2, ensure_ascii=False)
        
        # 更新缓存
        _config_cache[config_name] = new_config
        _config_versions[config_name] = new_version
        
        # 记录变更
        change = ConfigChange(
            config_name=config_name,
            version=new_version,
            changes=updates,
            changed_by=changed_by
        )
        
        # 触发监听器
        for watcher in _config_watchers:
            try:
                watcher(config_name, new_config)
            except Exception as e:
                logger.error(f"Config watcher error: {e}")
        
        logger.info(f"Config {config_name} updated: v{old_version} -> v{new_version}")
        
        return change
    
    @staticmethod
    def get_version(config_name: str) -> str:
        """获取配置版本"""
        return _config_versions.get(config_name, "unknown")
    
    @staticmethod
    def list_configs() -> Dict[str, Any]:
        """列出所有配置"""
        return {
            "configs": {name: {"version": _config_versions.get(name, "unknown")} 
                       for name in _config_cache.keys()},
            "total": len(_config_cache)
        }
    
    @staticmethod
    def register_watcher(callback: Callable):
        """注册配置变更监听器"""
        _config_watchers.append(callback)
    
    @staticmethod
    def start_file_watcher(interval: int = 30):
        """启动文件变更监听线程
        
        Args:
            interval: 检查间隔 (秒)
        """
        global _watch_thread, _running
        
        if _watch_thread is not None:
            return
        
        _running = True
        
        def watch_loop():
            while _running:
                try:
                    # 检查每个已加载配置的文件变更
                    for config_name in list(_config_cache.keys()):
                        current_version = _config_versions.get(config_name)
                        HotReloadManager.load_config(config_name)
                        new_version = _config_versions.get(config_name)
                        
                        if current_version and new_version and current_version != new_version:
                            logger.info(f"Config {config_name} changed on disk: v{current_version} -> v{new_version}")
                            for watcher in _config_watchers:
                                try:
                                    watcher(config_name, _config_cache[config_name])
                                except Exception as e:
                                    logger.error(f"Watcher error: {e}")
                except Exception as e:
                    logger.error(f"File watcher error: {e}")
                
                time.sleep(interval)
        
        _watch_thread = threading.Thread(target=watch_loop, daemon=True)
        _watch_thread.start()
        logger.info(f"Config file watcher started (interval={interval}s)")
    
    @staticmethod
    def stop_file_watcher():
        """停止文件监听"""
        global _running
        _running = False


# ========== 便捷函数 ==========

def reload_llm_providers() -> Dict[str, Any]:
    """热更新 LLM Provider 配置"""
    return HotReloadManager.get_config("llm_providers", reload=True)


def update_rate_limits(
    tenant_tier: str,
    rpm: Optional[int] = None,
    tpm: Optional[int] = None,
    concurrent: Optional[int] = None
) -> ConfigChange:
    """更新限流阈值
    
    Args:
        tenant_tier: 租户等级
        rpm: 每分钟请求数限制
        tpm: 每分钟 Token 数限制
        concurrent: 并发数限制
    
    Returns:
        变更记录
    """
    # 加载当前限流配置
    config_name = "rate_limits"
    current = HotReloadManager.get_config(config_name)
    
    # 如果配置不存在，创建默认结构
    if not current:
        current = {"tiers": {}}
    
    # 更新指定 tier
    if tenant_tier not in current.get("tiers", {}):
        current["tiers"][tenant_tier] = {}
    
    updates = {}
    if rpm is not None:
        current["tiers"][tenant_tier]["rpm"] = rpm
        updates["rpm"] = rpm
    if tpm is not None:
        current["tiers"][tenant_tier]["tpm"] = tpm
        updates["tpm"] = tpm
    if concurrent is not None:
        current["tiers"][tenant_tier]["concurrent"] = concurrent
        updates["concurrent"] = concurrent
    
    # 保存
    return HotReloadManager.update_config(config_name, {"tiers": current["tiers"]})


def enable_provider(provider_id: str, enabled: bool = True) -> ConfigChange:
    """启用/禁用 LLM Provider"""
    config_name = "llm_providers"
    current = HotReloadManager.get_config(config_name)
    
    if "providers" not in current:
        current["providers"] = []
    
    # 找到并更新 provider
    for p in current["providers"]:
        if p.get("provider_id") == provider_id:
            p["enabled"] = enabled
            break
    
    return HotReloadManager.update_config(config_name, {"providers": current["providers"]})


# 启动时加载核心配置
def init_configs():
    """初始化配置"""
    HotReloadManager.load_config("llm_providers")
    HotReloadManager.start_file_watcher(interval=60)
    logger.info("Hot reload manager initialized")