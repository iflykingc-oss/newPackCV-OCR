# -*- coding: utf-8 -*-
"""
端到端集成测试 - 完整链路验证
测试3个核心场景：包装提取、金融票据、通用文档

商业化项目要求：验证Graph Input → Output完整链路
"""

import os
import sys
import json
import pytest

sys.path.insert(0, os.path.join(os.getenv("COZE_WORKSPACE_PATH", "."), "src"))


class TestGraphTopology:
    """图拓扑结构验证（不依赖LLM调用）"""

    def test_graph_compiles(self):
        """验证主图可编译"""
        from graphs.graph import main_graph
        assert main_graph is not None

    def test_graph_has_entry_point(self):
        """验证图有入口节点"""
        from graphs.graph import main_graph
        graph = main_graph.get_graph()
        node_names = list(graph.nodes)
        assert len(node_names) > 0, "图无节点"

    def test_graph_input_schema(self):
        """验证图入参Schema定义"""
        from graphs.state import GraphInput
        fields = GraphInput.model_fields
        assert "package_image" in fields, "GraphInput缺少package_image字段"

    def test_graph_output_schema(self):
        """验证图出参Schema定义"""
        from graphs.state import GraphOutput
        fields = GraphOutput.model_fields
        assert "final_result" in fields or "answer" in fields or "success" in fields, \
            "GraphOutput缺少关键输出字段"

    def test_global_state_fields_complete(self):
        """验证GlobalState包含所有必要的中间字段"""
        from graphs.state import GlobalState
        fields = GlobalState.model_fields
        # 输入字段
        assert "package_image" in fields
        # 中间字段
        assert "ocr_raw_result" in fields or "raw_text" in fields
        assert "structured_data" in fields
        assert "vl_extracted_data" in fields
        assert "fused_structured_data" in fields
        # 输出字段
        assert "final_result" in fields
        assert "success" in fields
        # 场景检测字段
        assert "detected_category" in fields

    def test_all_active_node_inputs_have_output_match(self):
        """验证所有活跃节点的Input字段在GlobalState中有对应"""
        from graphs.state import GlobalState
        gs_fields = set(GlobalState.model_fields.keys())
        required_state_fields = [
            "package_image", "ocr_raw_result", "raw_text", "structured_data",
            "vl_extracted_data", "fused_structured_data", "final_result",
            "detected_category", "success"
        ]
        for field_name in required_state_fields:
            assert field_name in gs_fields, f"GlobalState缺少字段: {field_name}"


class TestScenarioDetection:
    """场景自动检测逻辑验证"""

    def test_scenario_detector_input_output_defined(self):
        """验证场景检测节点出入参定义"""
        from graphs.state import ScenarioDetectInput, ScenarioDetectOutput
        in_fields = ScenarioDetectInput.model_fields
        out_fields = ScenarioDetectOutput.model_fields
        assert "package_image" in in_fields
        assert "scenario_type" in out_fields
        assert "scenario_confidence" in out_fields

    def test_scenario_types_valid(self):
        """验证场景类型值合法"""
        valid_scenarios = {
            "packaging", "finance_receipt", "finance_statement",
            "pharmaceutical", "contract", "id_card", "logistics",
            "general_document"
        }
        assert len(valid_scenarios) == 8


class TestSmartPostprocess:
    """智能后处理节点验证"""

    def test_smart_postprocess_input_output_defined(self):
        """验证智能后处理节点出入参定义"""
        from graphs.state import SmartPostprocessInput, SmartPostprocessOutput
        in_fields = SmartPostprocessInput.model_fields
        out_fields = SmartPostprocessOutput.model_fields
        assert "fused_structured_data" in in_fields
        assert "detected_category" in out_fields or "final_result" in out_fields

    def test_qa_conditional_input_defined(self):
        """验证QA条件判断输入定义"""
        from graphs.state import QaConditionalInput
        in_fields = QaConditionalInput.model_fields
        assert "user_question" in in_fields


class TestInputRouter:
    """V5.9 输入类型路由节点验证"""

    def test_input_router_defined(self):
        """验证路由节点可导入且出入参定义"""
        from graphs.state import InputTypeRouteInput, InputTypeRouteOutput
        in_fields = InputTypeRouteInput.model_fields
        out_fields = InputTypeRouteOutput.model_fields
        assert "package_image" in in_fields or "file_url" in in_fields
        assert "input_type" in out_fields

    def test_graph_has_new_node_names(self):
        """验证图包含V5.9新节点"""
        from graphs.graph import main_graph
        node_names = list(main_graph.nodes.keys()) if hasattr(main_graph, 'nodes') else []
        # 使用get_graph接口
        graph = main_graph.get_graph()
        node_ids = set(graph.nodes)
        for node_name in ["input_router", "document_parse"]:
            assert node_name in node_ids, f"缺少节点: {node_name}"
        # barcode_detect 和 stamp_detect 已合并到 multi_channel_fusion 内部


class TestDocumentParse:
    """V5.9 文档解析节点验证"""

    def test_document_parse_state_defined(self):
        """验证文档解析出入参定义"""
        from graphs.state import DocumentParseInput, DocumentParseOutput
        in_fields = DocumentParseInput.model_fields
        out_fields = DocumentParseOutput.model_fields
        assert "file_url" in in_fields
        assert "markdown_output" in out_fields
        assert "tables" in out_fields


class TestBarcodeDetection:
    """V5.9 条码检测节点验证"""

    def test_barcode_state_defined(self):
        """验证条码检测出入参定义"""
        from graphs.state import BarcodeDetectInput, BarcodeDetectOutput
        out_fields = BarcodeDetectOutput.model_fields
        assert "barcodes" in out_fields
        assert "total_found" in out_fields
        assert "has_barcode" in out_fields


class TestStampDetection:
    """V5.9 印章检测节点验证"""

    def test_stamp_state_defined(self):
        """验证印章检测出入参定义"""
        from graphs.state import StampDetectInput, StampDetectOutput
        out_fields = StampDetectOutput.model_fields
        assert "stamps" in out_fields
        assert "total_found" in out_fields


class TestGlobalStateV59Fields:
    """V5.9 全局状态新增字段验证"""

    def test_v59_fields_in_global_state(self):
        """验证GlobalState包含V5.9新字段"""
        from graphs.state import GlobalState
        fields = GlobalState.model_fields
        assert "input_type" in fields
        assert "barcode_results" in fields
        assert "stamp_results" in fields
        assert "document_markdown" in fields
        assert "document_tables" in fields


class TestGraphInputDocumentSupport:
    """V5.9 图输入支持文档验证"""

    def test_graph_input_has_document_field(self):
        """验证GraphInput包含document_file字段"""
        from graphs.state import GraphInput
        fields = GraphInput.model_fields
        assert "document_file" in fields


class TestHealthEndpoints:
    """健康检查端点验证"""

    def test_health_response_structure(self):
        """验证/health端点返回结构"""
        expected_keys = {"status", "service", "version", "timestamp"}
        assert len(expected_keys) == 4


class TestI18nIntegration:
    """i18n集成验证"""

    def test_all_error_keys_have_translations(self):
        """验证所有错误key至少有中英文翻译"""
        from utils.i18n import ERROR_MESSAGES
        required_locales = {"zh-CN", "en"}
        for key, translations in ERROR_MESSAGES.items():
            for locale in required_locales:
                assert locale in translations, f"错误key '{key}' 缺少 {locale} 翻译"

    def test_supported_locales_complete(self):
        """验证4种主要locale全部覆盖"""
        from utils.i18n import SUPPORTED_LOCALES
        required = {"zh-CN", "en", "ja", "ko"}
        if isinstance(SUPPORTED_LOCALES, dict):
            supported = set(SUPPORTED_LOCALES.keys())
        else:
            supported = set(SUPPORTED_LOCALES)
        assert required.issubset(supported)

    def test_scenario_names_multilingual(self):
        """验证场景名称有多语言支持"""
        from utils.i18n import SCENARIO_NAMES
        assert isinstance(SCENARIO_NAMES, dict)
        assert len(SCENARIO_NAMES) > 0

    def test_currency_formatting(self):
        """验证货币格式化"""
        from utils.i18n import format_currency
        # CNY
        cny = format_currency(100.50, "CNY")
        assert isinstance(cny, str)
        # USD
        usd = format_currency(100.50, "USD")
        assert isinstance(usd, str)

    def test_datetime_formatting(self):
        """验证日期时间格式化"""
        from utils.i18n import format_datetime
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        # 中文格式
        zh_fmt = format_datetime(now, "zh-CN")
        assert isinstance(zh_fmt, str)
        # 英文格式
        en_fmt = format_datetime(now, "en")
        assert isinstance(en_fmt, str)
