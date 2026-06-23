# -*- coding: utf-8 -*-
"""
场景Schema注册表单元测试
验证8场景注册完整性、检测逻辑、Schema字段约束
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.getenv("COZE_WORKSPACE_PATH", "."), "src"))


class TestSchemaRegistry:
    """SchemaRegistry 核心功能测试"""

    def setup_method(self):
        """每个测试方法前初始化"""
        from utils.scenario_schemas.registry import SchemaRegistry
        self.registry = SchemaRegistry()

    def test_all_8_scenarios_registered(self):
        """验证8个场景全部注册"""
        expected = {
            "packaging", "finance_receipt", "finance_statement",
            "pharmaceutical", "contract", "id_card", "logistics",
            "general_document"
        }
        all_schemas = self.registry.get_all()
        registered = set(all_schemas.keys())
        assert expected == registered, f"缺失场景: {expected - registered}, 多余场景: {registered - expected}"

    def test_get_schema_packaging(self):
        """验证packaging场景Schema可获取"""
        schema = self.registry.get("packaging")
        assert schema is not None
        assert schema.scenario_type == "packaging"
        required = schema.get_required_fields()
        assert len(required) >= 2  # 至少2个必填字段
        assert len(schema.fields) >= 9  # 至少9个总字段

    def test_get_schema_finance_receipt(self):
        """验证finance_receipt场景Schema"""
        schema = self.registry.get("finance_receipt")
        assert schema is not None
        assert schema.scenario_type == "finance_receipt"
        assert len(schema.get_required_fields()) >= 6

    def test_get_schema_finance_statement(self):
        """验证finance_statement场景Schema"""
        schema = self.registry.get("finance_statement")
        assert schema is not None
        assert schema.scenario_type == "finance_statement"

    def test_get_schema_pharmaceutical(self):
        """验证pharmaceutical场景Schema"""
        schema = self.registry.get("pharmaceutical")
        assert schema is not None
        assert schema.scenario_type == "pharmaceutical"

    def test_get_schema_contract(self):
        """验证contract场景Schema"""
        schema = self.registry.get("contract")
        assert schema is not None
        assert schema.scenario_type == "contract"

    def test_get_schema_id_card(self):
        """验证id_card场景Schema"""
        schema = self.registry.get("id_card")
        assert schema is not None
        assert schema.scenario_type == "id_card"

    def test_get_schema_logistics(self):
        """验证logistics场景Schema"""
        schema = self.registry.get("logistics")
        assert schema is not None
        assert schema.scenario_type == "logistics"

    def test_get_schema_general(self):
        """验证general_document场景Schema"""
        schema = self.registry.get("general_document")
        assert schema is not None
        assert schema.scenario_type == "general_document"

    def test_get_nonexistent_schema_returns_general(self):
        """验证不存在的场景回退到general_document"""
        schema = self.registry.get("nonexistent_scenario")
        assert schema is not None
        assert schema.scenario_type == "general_document"

    def test_detect_scenario_keywords_packaging(self):
        """关键词检测：包装场景"""
        result = self.registry.detect_scenario("产品配料表 生产日期 保质期 净含量")
        assert result in ("packaging", "pharmaceutical")

    def test_detect_scenario_keywords_finance(self):
        """关键词检测：金融场景"""
        result = self.registry.detect_scenario("银行回单 收款人 金额 发票 收据")
        assert result in ("finance_receipt", "finance_statement")

    def test_detect_scenario_keywords_contract(self):
        """关键词检测：合同场景"""
        result = self.registry.detect_scenario("甲方 乙方 合同编号 违约责任")
        assert result == "contract"

    def test_detect_scenario_keywords_id_card(self):
        """关键词检测：证件场景"""
        result = self.registry.detect_scenario("身份证号码 有效期限 签发机关 居民身份证")
        assert result == "id_card"

    def test_detect_scenario_keywords_logistics(self):
        """关键词检测：物流场景"""
        result = self.registry.detect_scenario("运单号 寄件人 收件人 快递 物流")
        assert result == "logistics"

    def test_detect_scenario_no_match_defaults_general(self):
        """关键词无法匹配时默认general_document"""
        result = self.registry.detect_scenario("这是一段无关文本")
        assert result == "general_document"

    def test_schema_fields_have_descriptions(self):
        """验证所有场景的字段都有description"""
        for scenario_name, schema in self.registry.get_all().items():
            for field_def in schema.fields:
                assert field_def.description, f"场景{scenario_name}的字段{field_def.name}缺少description"

    def test_schema_field_types_valid(self):
        """验证所有字段类型都是合法类型"""
        valid_types = {"str", "int", "float", "bool", "list", "dict", "date", "datetime", "number"}
        for scenario_name, schema in self.registry.get_all().items():
            for field_def in schema.fields:
                assert field_def.field_type in valid_types, \
                    f"场景{scenario_name}的字段{field_def.name}类型{field_def.field_type}不合法"

    def test_schema_has_system_prompt(self):
        """验证每个场景都有LLM系统提示词"""
        for scenario_name, schema in self.registry.get_all().items():
            assert schema.system_prompt, f"场景{scenario_name}缺少system_prompt"
            assert len(schema.system_prompt) > 20, f"场景{scenario_name}的system_prompt过短"

    def test_schema_has_user_prompt_template(self):
        """验证每个场景都有用户提示词模板"""
        for scenario_name, schema in self.registry.get_all().items():
            assert schema.user_prompt_template, f"场景{scenario_name}缺少user_prompt_template"

    def test_list_scenarios(self):
        """验证场景列表API"""
        scenarios = self.registry.list_scenarios()
        assert len(scenarios) == 8
        for s in scenarios:
            assert "type" in s
            assert "name" in s
            assert "required" in s
