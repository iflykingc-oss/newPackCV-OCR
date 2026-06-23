# -*- coding: utf-8 -*-
"""
PackCV 统一配置管理器
三级配置源：文件默认 → 数据库租户 → 运行时注入
优先级：运行时 > 租户 > 文件默认
"""

import os
import json
import logging
import sqlite3
import threading
from typing import Dict, Any, Optional, List
from datetime import datetime

# 场景→LLM配置映射表
_SCENARIO_LLM_MAP: Dict[str, str] = {
    "packaging": "model_extract",
    "finance_receipt": "finance_extract",
    "finance_statement": "finance_statement",
    "pharmaceutical": "pharma_extract",
    "contract": "contract_extract",
    "id_card": "id_card_extract",
    "logistics": "logistics_extract",
    "general_document": "general_extract",
}

logger = logging.getLogger("config_manager")

_DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "assets", "packcv_config.db"
)

# ─── 默认配置文件索引 ───────────────────────────────────────────
_CONFIG_FILE_INDEX = {
    "llm": {
        "model_extract": "config/model_extract_llm_cfg.json",
        "vl_packaging": "config/vl_packaging_llm_cfg.json",
        "correct_text": "config/correct_text_llm_cfg.json",
        "qa_answer": "config/qa_answer_llm_cfg.json",
        "knowledge_inference": "config/knowledge_inference_llm_cfg.json",
    },
    "engine": "src/config/engine_adapter_cfg.json",
}

# ─── 默认配置模板（当文件不存在时使用） ─────────────────────────
_DEFAULT_CONFIGS = {
    "llm": {
        "model_extract": {
            "model": "doubao-seed-1.5", "temperature": 0.0, "max_tokens": 4000
        },
        "vl_packaging": {
            "model": "doubao-seed-1.5-vl", "temperature": 0.0, "max_tokens": 4000
        },
        "correct_text": {
            "model": "doubao-seed-1.5", "temperature": 0.0, "max_tokens": 2000
        },
        "qa_answer": {
            "model": "doubao-seed-1.5", "temperature": 0.1, "max_tokens": 2000
        },
        "knowledge_inference": {
            "model": "doubao-seed-1.5", "temperature": 0.0, "max_tokens": 2000
        },
    },
    "engine": {
        "ocr": {"engine_type": "builtin"},
        "vl": {"enabled": True},
    },
}


class ConfigManager:
    """三级配置中心（单例）"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, db_path: Optional[str] = None):
        if hasattr(self, "_initialized") and self._initialized:
            return
        self._db_path = db_path or _DEFAULT_DB_PATH
        self._default_configs: Dict[str, Any] = {}   # 文件默认
        self._runtime_overrides: Dict[str, Any] = {}  # 运行时临时覆盖
        self._initialized = True
        self._load_defaults()
        self._init_db()

    # ── 加载默认配置 ────────────────────────────────────────────

    def _load_defaults(self):
        """从配置文件加载默认值"""
        ws = os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects")

        # 加载LLM配置
        llm_configs = {}
        for node_id, rel_path in _CONFIG_FILE_INDEX["llm"].items():
            full_path = os.path.join(ws, rel_path)
            cfg = {}
            if os.path.exists(full_path):
                try:
                    with open(full_path, "r") as f:
                        data = json.load(f)
                    cfg = {
                        "model": data.get("config", {}).get("model", ""),
                        "temperature": data.get("config", {}).get("temperature", 0.0),
                        "max_tokens": data.get("config", {}).get("max_completion_tokens", 2000),
                        "sp": data.get("sp", ""),
                        "up": data.get("up", ""),
                    }
                except Exception as e:
                    logger.warning(f"加载配置失败 {rel_path}: {e}")
            else:
                cfg = _DEFAULT_CONFIGS["llm"].get(node_id, {})
                logger.info(f"配置文件不存在，使用默认模板: {rel_path}")
            llm_configs[node_id] = cfg

        # 加载引擎配置
        engine_path = os.path.join(ws, _CONFIG_FILE_INDEX["engine"])
        engine_config = {}
        if os.path.exists(engine_path):
            try:
                with open(engine_path, "r") as f:
                    engine_config = json.load(f)
            except Exception as e:
                logger.warning(f"加载引擎配置失败: {e}")

        self._default_configs = {
            "llm": llm_configs,
            "engine": engine_config or _DEFAULT_CONFIGS["engine"],
        }
        logger.info(f"默认配置加载完成: {len(llm_configs)}个LLM配置, 引擎配置已加载")

    # ── 数据库 ──────────────────────────────────────────────────

    def _init_db(self):
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        conn = sqlite3.connect(self._db_path)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS tenant_configs (
                tenant_id TEXT PRIMARY KEY,
                config_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                version INTEGER DEFAULT 1,
                is_active INTEGER DEFAULT 1
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS config_templates (
                template_name TEXT PRIMARY KEY,
                description TEXT DEFAULT '',
                config_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()
        logger.info(f"配置数据库就绪: {self._db_path}")

    def _get_db(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    # ── 租户配置 CRUD ──────────────────────────────────────────

    def set_tenant_config(self, tenant_id: str, config: Dict[str, Any]) -> bool:
        """设置/更新租户配置"""
        now = datetime.now().isoformat()
        config_json = json.dumps(config, ensure_ascii=False)
        conn = self._get_db()
        try:
            c = conn.cursor()
            c.execute("""
                INSERT INTO tenant_configs (tenant_id, config_json, created_at, updated_at, version)
                VALUES (?, ?, ?, ?, 1)
                ON CONFLICT(tenant_id) DO UPDATE SET
                    config_json=excluded.config_json,
                    updated_at=excluded.updated_at,
                    version=tenant_configs.version + 1
            """, (tenant_id, config_json, now, now))
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"设置租户配置失败 [{tenant_id}]: {e}")
            return False
        finally:
            conn.close()

    def get_tenant_config(self, tenant_id: str) -> Optional[Dict[str, Any]]:
        """获取租户配置"""
        conn = self._get_db()
        try:
            c = conn.cursor()
            c.execute("SELECT config_json FROM tenant_configs WHERE tenant_id=? AND is_active=1", (tenant_id,))
            row = c.fetchone()
            if row:
                return json.loads(row[0])
            return None
        except Exception as e:
            logger.error(f"获取租户配置失败 [{tenant_id}]: {e}")
            return None
        finally:
            conn.close()

    def delete_tenant_config(self, tenant_id: str) -> bool:
        """软删除租户配置"""
        conn = self._get_db()
        try:
            c = conn.cursor()
            c.execute("UPDATE tenant_configs SET is_active=0, updated_at=? WHERE tenant_id=?", 
                      (datetime.now().isoformat(), tenant_id))
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"删除租户配置失败 [{tenant_id}]: {e}")
            return False
        finally:
            conn.close()

    def list_tenant_configs(self, include_inactive: bool = False) -> List[Dict[str, Any]]:
        """列出所有租户配置"""
        conn = self._get_db()
        try:
            c = conn.cursor()
            if include_inactive:
                c.execute("SELECT * FROM tenant_configs ORDER BY updated_at DESC")
            else:
                c.execute("SELECT * FROM tenant_configs WHERE is_active=1 ORDER BY updated_at DESC")
            rows = c.fetchall()
            return [
                {"tenant_id": r[0], "config": json.loads(r[1]), 
                 "created_at": r[2], "updated_at": r[3], "version": r[4]}
                for r in rows
            ]
        finally:
            conn.close()

    # ── 运行时覆盖 ──────────────────────────────────────────────

    def set_runtime_override(self, overrides: Dict[str, Any]):
        """设置运行时覆盖（合并而非替换）"""
        self._runtime_overrides = self._deep_merge(self._runtime_overrides, overrides)

    def clear_runtime_override(self):
        """清除运行时覆盖"""
        self._runtime_overrides = {}

    # ── 三级解析 ────────────────────────────────────────────────

    def resolve(self, tenant_id: Optional[str] = None,
                runtime_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        三级解析配置
        优先级：runtime_config > tenant_config > default_config
        
        Args:
            tenant_id: 租户ID（可选）
            runtime_config: 运行时注入配置（可选）
        Returns:
            合并后的完整配置
        """
        # Level 3: 文件默认
        config = self._deep_merge({}, self._default_configs)

        # Level 2: 租户覆盖（DB持久化）
        if tenant_id:
            tenant_cfg = self.get_tenant_config(tenant_id)
            if tenant_cfg:
                config = self._deep_merge(config, tenant_cfg)

        # Level 1: 运行时注入（最高优先级）
        if runtime_config:
            config = self._deep_merge(config, runtime_config)

        # 合并运行时override
        if self._runtime_overrides:
            config = self._deep_merge(config, self._runtime_overrides)

        return config

    def resolve_llm_config(self, node_id: str, tenant_id: Optional[str] = None,
                           runtime_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        解析单个LLM节点的配置（快捷方法）
        
        搜索路径:
        1. runtime_config.llm.{node_id}
        2. tenant_config.llm.{node_id}
        3. default_config.llm.{node_id}
        """
        full = self.resolve(tenant_id, runtime_config)
        return full.get("llm", {}).get(node_id, {})

    def resolve_scenario_config(self, scenario_type: str, tenant_id: Optional[str] = None,
                                  runtime_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        解析场景级LLM配置（三级配置链）
        
        将场景类型映射到对应的LLM配置节点，然后走三级解析。
        scenario_type → node_id → file(默认) → tenant(DB) → runtime(注入)
        
        Args:
            scenario_type: 场景类型（packaging/finance_receipt/...）
            tenant_id: 租户ID（可选）
            runtime_config: 运行时注入配置（可选）
        Returns:
            解析后的LLM配置（含model/config/sp/up）
        """
        node_id = _SCENARIO_LLM_MAP.get(scenario_type, "general_extract")
        return self.resolve_llm_config(node_id, tenant_id, runtime_config)

    # ── 工具方法 ────────────────────────────────────────────────

    @staticmethod
    def _deep_merge(base: Dict, override: Dict) -> Dict:
        """深度合并两个字典（override的键值覆盖base）"""
        result = dict(base)
        for key, val in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(val, dict):
                result[key] = ConfigManager._deep_merge(result[key], val)
            else:
                result[key] = val
        return result

    def get_all_nodes(self) -> List[Dict[str, str]]:
        """获取所有可配置的节点清单"""
        return [
            {"node_id": "model_extract", "name": "结构化提取", "type": "LLM"},
            {"node_id": "vl_packaging", "name": "VL多模态理解", "type": "LLM"},
            {"node_id": "correct_text", "name": "文本纠错", "type": "LLM"},
            {"node_id": "qa_answer", "name": "智能问答", "type": "LLM"},
            {"node_id": "knowledge_inference", "name": "知识推理", "type": "LLM"},
            {"node_id": "ocr_engine", "name": "OCR引擎选择", "type": "engine"},
            {"node_id": "vl_engine", "name": "VL引擎选择", "type": "engine"},
            {"node_id": "quality_enhance", "name": "画质增强", "type": "engine"},
            {"node_id": "curvature_correct", "name": "弯曲文本校正", "type": "engine"},
        ]

    def get_config_summary(self, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """获取配置摘要（用于IM卡片展示）"""
        config = self.resolve(tenant_id)
        llm = config.get("llm", {})
        engine = config.get("engine", {})
        return {
            "tenant_id": tenant_id or "global",
            "models": {
                node_id: cfg.get("model", "default")
                for node_id, cfg in llm.items()
            },
            "ocr_engine": engine.get("ocr", {}).get("engine_type", "builtin"),
            "vl_enabled": engine.get("vl", {}).get("enabled", True),
        }