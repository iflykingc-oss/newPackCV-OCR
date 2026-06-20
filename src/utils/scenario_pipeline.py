#!/usr/bin/env python3
"""场景Pipeline工厂：根据检测到的场景类型输出抽取配置"""
import os, json
from typing import Dict, Any, Optional
from utils.scenario_schemas.registry import SchemaRegistry, default_registry


class ScenarioPipeline:
    """场景流水线工厂 - 生产场景专用的提取配置"""

    SCENARIO_PROMPTS: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def _load_prompts(cls) -> Dict[str, Any]:
        """从配置文件加载各场景的SP/UP"""
        if cls.SCENARIO_PROMPTS:
            return cls.SCENARIO_PROMPTS
        cfg_dir = os.path.join(os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects"), "config")
        scenario_files = {
            "packaging": "model_extract_llm_cfg.json",
            "finance_receipt": "finance_extract_llm_cfg.json",
            "finance_statement": "finance_statement_llm_cfg.json",
            "pharmaceutical": "pharma_extract_llm_cfg.json",
            "general_document": "general_extract_llm_cfg.json",
        }
        for scenario, fname in scenario_files.items():
            fpath = os.path.join(cfg_dir, fname)
            try:
                with open(fpath) as f:
                    cls.SCENARIO_PROMPTS[scenario] = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                # 降级：使用Schema中的SP
                schema = default_registry.get(scenario)
                if schema:
                    cls.SCENARIO_PROMPTS[scenario] = {
                        "sp": schema.system_prompt,
                        "up": schema.user_prompt_template,
                    }
        return cls.SCENARIO_PROMPTS

    @classmethod
    def get_scenario_config(cls, scenario_type: str) -> Dict[str, Any]:
        """获取场景配置（SP/UP/字段列表/预处理参数）"""
        prompts = cls._load_prompts()
        config = prompts.get(scenario_type, prompts.get("general_document", {}))
        config["scenario_type"] = scenario_type
        schema = default_registry.get(scenario_type)
        if schema:
            config["fields"] = [{"name": f.name, "required": f.required, "type": f.field_type or "str"} for f in schema.fields]
        return config

    @classmethod
    def get_preprocess_params(cls, scenario_type: str) -> Dict[str, Any]:
        """获取场景专用预处理参数"""
        defaults = {
            "clahe_enabled": True,
            "deblur_enabled": True,
            "perspective_correct": False,
            "curved_text_correct": False,
            "tps_strength": 0.5,
            "enhance_resolution": False,
        }
        scenario_overrides = {
            "packaging": {
                "curved_text_correct": True,
                "tps_strength": 0.7,
            },
            "finance_receipt": {
                "perspective_correct": True,
                "clahe_enabled": True,
                "enhance_resolution": True,
            },
            "finance_statement": {
                "perspective_correct": True,
                "clahe_enabled": True,
                "enhance_resolution": True,
                "deblur_enabled": True,
            },
            "pharmaceutical": {
                "curved_text_correct": True,
                "tps_strength": 0.8,
                "enhance_resolution": True,
            },
            "general_document": {
                "perspective_correct": True,
                "clahe_enabled": True,
            },
        }
        result = dict(defaults)
        result.update(scenario_overrides.get(scenario_type, {}))
        return result