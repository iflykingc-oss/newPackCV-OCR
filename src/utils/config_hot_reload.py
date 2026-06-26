#!/usr/bin/env python3
"""配置热更新管理器 - 支持动态加载和监听配置变化"""
import os
import json
import logging
import time
import threading
from typing import Dict, Any, Optional, Callable
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfigCache:
    """配置缓存，存储已加载的配置"""
    
    def __init__(self):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.timestamps: Dict[str, float] = {}
        self.max_age_seconds: int = 300  # 配置最大缓存时间5分钟
    
    def get(self, config_path: str) -> Optional[Dict[str, Any]]:
        """获取缓存的配置"""
        if config_path in self.cache:
            # 检查缓存是否过期
            if time.time() - self.timestamps[config_path] < self.max_age_seconds:
                return self.cache[config_path]
            else:
                # 清除过期缓存
                self.invalidate(config_path)
        return None
    
    def set(self, config_path: str, config: Dict[str, Any]) -> None:
        """设置配置缓存"""
        self.cache[config_path] = config
        self.timestamps[config_path] = time.time()
    
    def invalidate(self, config_path: str) -> None:
        """清除指定配置缓存"""
        if config_path in self.cache:
            del self.cache[config_path]
        if config_path in self.timestamps:
            del self.timestamps[config_path]
    
    def clear_all(self) -> None:
        """清除所有缓存"""
        self.cache.clear()
        self.timestamps.clear()


class ConfigHotLoader:
    """配置热加载器"""
    
    def __init__(self, config_dir: Optional[str] = None):
        self.config_dir: str = config_dir or os.path.join(
            os.getenv("COZE_WORKSPACE_PATH", ""), "config"
        )
        self.cache: ConfigCache = ConfigCache()
        self.file_watchers: Dict[str, float] = {}  # 文件路径 -> 上次修改时间
        self.reload_callbacks: Dict[str, Callable] = {}  # 配置路径 -> 回调函数
    
    def load_config(self, config_path: str, force_reload: bool = False) -> Dict[str, Any]:
        """加载配置文件"""
        # 构建完整路径
        if not os.path.isabs(config_path):
            full_path: str = os.path.join(self.config_dir, config_path)
        else:
            full_path = config_path
        
        # 检查缓存
        if not force_reload:
            cached: Optional[Dict[str, Any]] = self.cache.get(full_path)
            if cached is not None:
                logger.info(f"使用缓存配置: {config_path}")
                return cached
        
        # 从文件加载
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                config: Dict[str, Any] = json.load(f)
            
            # 更新缓存
            self.cache.set(full_path, config)
            self.file_watchers[full_path] = os.path.getmtime(full_path)
            
            logger.info(f"加载配置成功: {config_path}")
            return config
            
        except FileNotFoundError:
            logger.warning(f"配置文件不存在: {full_path}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"配置文件JSON解析失败: {full_path}, {e}")
            return {}
        except Exception as e:
            logger.error(f"加载配置失败: {full_path}, {e}")
            return {}
    
    def register_reload_callback(self, config_path: str, callback: Callable) -> None:
        """注册配置重加载回调"""
        full_path: str = os.path.join(self.config_dir, config_path) if not os.path.isabs(config_path) else config_path
        self.reload_callbacks[full_path] = callback
        logger.info(f"注册配置重加载回调: {config_path}")
    
    def check_and_reload(self) -> Dict[str, bool]:
        """检查配置文件变化并重加载"""
        reload_results: Dict[str, bool] = {}
        
        for full_path, last_mtime in self.file_watchers.items():
            try:
                current_mtime: float = os.path.getmtime(full_path)
                
                if current_mtime > last_mtime:
                    logger.info(f"检测到配置文件变化: {full_path}")
                    
                    # 清除缓存并重新加载
                    self.cache.invalidate(full_path)
                    new_config: Dict[str, Any] = self.load_config(full_path, force_reload=True)
                    
                    # 更新文件修改时间
                    self.file_watchers[full_path] = current_mtime
                    
                    # 调用回调函数
                    if full_path in self.reload_callbacks:
                        try:
                            self.reload_callbacks[full_path](new_config)
                            reload_results[full_path] = True
                            logger.info(f"配置重加载回调执行成功: {full_path}")
                        except Exception as e:
                            reload_results[full_path] = False
                            logger.error(f"配置重加载回调执行失败: {full_path}, {e}")
                    else:
                        reload_results[full_path] = True
                        
            except Exception as e:
                logger.error(f"检查配置文件变化失败: {full_path}, {e}")
                reload_results[full_path] = False
        
        return reload_results
    
    def start_background_watcher(self, interval_seconds: int = 30) -> threading.Thread:
        """启动后台配置监听线程"""
        def watch_loop():
            while True:
                try:
                    self.check_and_reload()
                    time.sleep(interval_seconds)
                except Exception as e:
                    logger.error(f"配置监听线程异常: {e}")
                    time.sleep(interval_seconds)
        
        thread: threading.Thread = threading.Thread(target=watch_loop, daemon=True)
        thread.start()
        logger.info(f"启动配置监听线程, interval={interval_seconds}s")
        return thread
    
    def get_llm_config(self, config_path: str) -> Dict[str, Any]:
        """获取LLM配置"""
        config: Dict[str, Any] = self.load_config(config_path)
        
        return {
            "model": config.get("config", {}).get("model", "doubao-seed-1-8-251228"),
            "temperature": config.get("config", {}).get("temperature", 0.7),
            "top_p": config.get("config", {}).get("top_p", 0.7),
            "max_completion_tokens": config.get("config", {}).get("max_completion_tokens", 32768),
            "sp": config.get("sp", ""),
            "up": config.get("up", ""),
            "tools": config.get("tools", []),
        }


# 全局配置加载器
_config_loader: Optional[ConfigHotLoader] = None


def get_config_loader() -> ConfigHotLoader:
    """获取全局配置加载器"""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigHotLoader()
    return _config_loader


def load_llm_config(config_path: str, force_reload: bool = False) -> Dict[str, Any]:
    """快捷加载LLM配置"""
    loader: ConfigHotLoader = get_config_loader()
    return loader.get_llm_config(config_path)