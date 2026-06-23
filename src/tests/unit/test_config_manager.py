# -*- coding: utf-8 -*-
"""
ConfigManager单元测试
验证三级配置链、租户配置、场景配置解析
"""

import os
import sys
import json
import pytest
import time

sys.path.insert(0, os.path.join(os.getenv("COZE_WORKSPACE_PATH", "."), "src"))


def _unique_tenant(prefix: str) -> str:
    """生成唯一tenant_id避免测试间冲突"""
    return f"{prefix}_{int(time.time() * 1000)}"


class TestConfigManager:
    """ConfigManager 三级配置链测试"""

    def setup_method(self):
        """每个测试前初始化"""
        from utils.config_manager import ConfigManager
        self.cm = ConfigManager()

    def test_default_config_loaded(self):
        """验证默认配置文件加载"""
        resolved = self.cm.resolve()
        assert isinstance(resolved, dict)
        assert len(resolved) > 0

    def test_tenant_config_set_and_get(self):
        """设置和获取租户配置"""
        tenant_id = _unique_tenant("test_set")
        test_config = {
            "ocr_engine_type": "smart",
            "model_name": "test-model"
        }
        success = self.cm.set_tenant_config(tenant_id, test_config)
        assert success, "设置租户配置失败"
        retrieved = self.cm.get_tenant_config(tenant_id)
        assert retrieved is not None
        assert retrieved.get("ocr_engine_type") == "smart"
        # 清理
        self.cm.delete_tenant_config(tenant_id)

    def test_tenant_config_override(self):
        """租户配置覆盖默认配置"""
        tenant_id = _unique_tenant("test_override")
        override = {"engine": {"ocr": {"engine_type": "rapidocr"}}}
        self.cm.set_tenant_config(tenant_id, override)
        resolved = self.cm.resolve(tenant_id)
        # 租户配置应覆盖默认值
        ocr_type = resolved.get("engine", {}).get("ocr", {}).get("engine_type")
        assert ocr_type == "rapidocr"
        # 清理
        self.cm.delete_tenant_config(tenant_id)

    def test_runtime_override(self):
        """运行时注入覆盖"""
        self.cm.set_runtime_override({"model_name": "runtime-model"})
        resolved = self.cm.resolve()
        assert resolved.get("model_name") == "runtime-model"
        # 清理
        self.cm.clear_runtime_override()

    def test_tenant_config_delete(self):
        """删除租户配置"""
        tenant_id = _unique_tenant("test_delete")
        self.cm.set_tenant_config(tenant_id, {"ocr_engine_type": "tesseract"})
        # 验证设置成功
        got = self.cm.get_tenant_config(tenant_id)
        assert got is not None, "租户配置设置后应能获取"
        # 删除
        self.cm.delete_tenant_config(tenant_id)
        # 验证删除后为None
        result = self.cm.get_tenant_config(tenant_id)
        assert result is None, "删除后应返回None"

    def test_config_summary(self):
        """配置摘要生成"""
        summary = self.cm.get_config_summary()
        assert isinstance(summary, dict)

    def test_config_summary_with_tenant(self):
        """带租户的配置摘要"""
        tenant_id = _unique_tenant("test_summary")
        self.cm.set_tenant_config(tenant_id, {"model_name": "summary-model"})
        summary = self.cm.get_config_summary(tenant_id)
        assert isinstance(summary, dict)
        # 清理
        self.cm.delete_tenant_config(tenant_id)

    def test_scenario_config_resolution(self):
        """场景级配置解析"""
        from utils.scenario_schemas.registry import SchemaRegistry
        reg = SchemaRegistry()
        for scenario_name in reg.get_all().keys():
            resolved = self.cm.resolve_scenario_config(scenario_name)
            assert isinstance(resolved, dict), f"场景{scenario_name}配置解析失败"

    def test_three_level_chain_priority(self):
        """三级优先级验证：runtime > tenant > file"""
        tenant_id = _unique_tenant("test_priority")
        # Level 2: tenant override
        self.cm.set_tenant_config(tenant_id, {"engine": {"ocr": {"engine_type": "paddleocr"}}})
        tenant_resolved = self.cm.resolve(tenant_id)
        assert tenant_resolved.get("engine", {}).get("ocr", {}).get("engine_type") == "paddleocr"
        # Level 1: runtime override (highest)
        self.cm.set_runtime_override({"engine": {"ocr": {"engine_type": "smart"}}})
        runtime_resolved = self.cm.resolve(tenant_id)
        assert runtime_resolved.get("engine", {}).get("ocr", {}).get("engine_type") == "smart"
        # 清理
        self.cm.clear_runtime_override()
        self.cm.delete_tenant_config(tenant_id)

    def test_list_tenants(self):
        """列出租户配置"""
        tenants = self.cm.list_tenant_configs()
        assert isinstance(tenants, list)

    def test_resolve_llm_config(self):
        """解析LLM节点配置"""
        llm_cfg = self.cm.resolve_llm_config("model_extract")
        assert isinstance(llm_cfg, dict)
