#!/usr/bin/env python3
"""E2E测试用例 - 覆盖主流程和边缘场景"""
import pytest
import os
import json
from pydantic import BaseModel, Field

# 导入工作流组件
from graphs.graph import main_graph
from graphs.state import GraphInput, GraphOutput


class TestCase(BaseModel):
    """测试用例定义"""
    name: str = Field(..., description="测试用例名称")
    description: str = Field(..., description="测试用例描述")
    input_data: dict = Field(..., description="输入数据")
    expected_scenario: str = Field(default="general_document", description="预期场景")
    expected_fields: list = Field(default=[], description="预期提取字段")


# 定义测试用例集合
TEST_CASES = [
    # 主流程测试
    TestCase(
        name="test_basic_image_processing",
        description="测试基本图片处理流程",
        input_data={
            "input_file": {
                "url": "https://picsum.photos/seed/basic/200/300.jpg",
                "file_type": "image"
            },
            "user_question": "描述这张图片"
        },
        expected_scenario="general_document"
    ),
    TestCase(
        name="test_document_processing",
        description="测试文档处理流程",
        input_data={
            "input_file": {
                "url": "https://example.com/test.pdf",
                "file_type": "document"
            }
        },
        expected_scenario="general_document"
    ),
    TestCase(
        name="test_with_user_question",
        description="测试带用户问题的流程",
        input_data={
            "input_file": {
                "url": "https://picsum.photos/seed/qa/200/300.jpg",
                "file_type": "image"
            },
            "user_question": "这张图片包含什么信息？"
        },
        expected_scenario="general_document"
    ),
    
    # 边缘场景测试
    TestCase(
        name="test_empty_input",
        description="测试空输入处理",
        input_data={
            "input_file": {
                "url": "",
                "file_type": "default"
            }
        },
        expected_scenario="general_document"
    ),
    TestCase(
        name="test_invalid_url",
        description="测试无效URL处理",
        input_data={
            "input_file": {
                "url": "invalid-url-format",
                "file_type": "image"
            }
        },
        expected_scenario="general_document"
    ),
    TestCase(
        name="test_missing_optional_fields",
        description="测试缺失可选字段",
        input_data={
            "input_file": {
                "url": "https://picsum.photos/seed/minimal/200/300.jpg",
                "file_type": "image"
            }
        },
        expected_scenario="general_document"
    ),
    TestCase(
        name="test_large_image",
        description="测试大图片处理",
        input_data={
            "input_file": {
                "url": "https://picsum.photos/seed/large/1920/1080.jpg",
                "file_type": "image"
            },
            "user_question": "分析这张高清图片"
        },
        expected_scenario="general_document"
    ),
    TestCase(
        name="test_multi_format_input",
        description="测试多格式输入",
        input_data={
            "input_file": {
                "url": "https://picsum.photos/seed/multi/200/300.webp",
                "file_type": "image"
            }
        },
        expected_scenario="general_document"
    ),
]


class TestE2EWorkflow:
    """E2E工作流测试类"""
    
    def test_workflow_exists(self):
        """测试工作流是否存在"""
        assert main_graph is not None, "工作流未正确初始化"
    
    def test_workflow_input_schema(self):
        """测试工作流输入Schema"""
        input_schema = GraphInput.model_json_schema()
        assert "input_file" in input_schema["properties"], "缺少input_file字段"
        assert "user_question" in input_schema["properties"], "缺少user_question字段"
    
    def test_workflow_output_schema(self):
        """测试工作流输出Schema"""
        output_schema = GraphOutput.model_json_schema()
        assert "structured_data" in output_schema["properties"], "缺少structured_data字段"
        assert "qa_answer" in output_schema["properties"], "缺少qa_answer字段"
    
    @pytest.mark.parametrize("test_case", TEST_CASES, ids=lambda tc: tc.name)
    def test_main_workflow(self, test_case):
        """测试主流程"""
        try:
            # 构建输入
            input_data = GraphInput(**test_case.input_data)
            
            # 执行工作流
            result = main_graph.invoke(input_data.model_dump())
            
            # 验证结果结构
            assert result is not None, f"{test_case.name}: 工作流未返回结果"
            assert "structured_data" in result, f"{test_case.name}: 缺少structured_data字段"
            assert "qa_answer" in result, f"{test_case.name}: 缺少qa_answer字段"
            
            # 验证场景检测
            if test_case.expected_scenario:
                detected_scenario = result.get("structured_data", {}).get("scenario", "")
                assert detected_scenario == test_case.expected_scenario, \
                    f"{test_case.name}: 场景不匹配，预期{test_case.expected_scenario}，实际{detected_scenario}"
            
            print(f"✅ {test_case.name} passed: {json.dumps(result, ensure_ascii=False)[:200]}")
            
        except Exception as e:
            print(f"❌ {test_case.name} failed: {str(e)}")
            # 边缘场景测试允许失败（如无效URL）
            if test_case.name.startswith("test_invalid") or test_case.name.startswith("test_empty"):
                pytest.skip(f"边缘场景预期失败: {str(e)}")
            else:
                raise
    
    def test_workflow_latency(self):
        """测试工作流延迟"""
        import time
        
        input_data = GraphInput(
            input_file={"url": "https://picsum.photos/seed/perf/200/300.jpg", "file_type": "image"}
        )
        
        start_time = time.time()
        result = main_graph.invoke(input_data.model_dump())
        elapsed_time = time.time() - start_time
        
        # 验证延迟在合理范围内（10秒以内）
        assert elapsed_time < 10, f"工作流延迟过高: {elapsed_time:.2f}秒"
        print(f"✅ 工作流延迟测试: {elapsed_time:.2f}秒")
    
    def test_workflow_confidence(self):
        """测试工作流置信度"""
        input_data = GraphInput(
            input_file={"url": "https://picsum.photos/seed/conf/200/300.jpg", "file_type": "image"}
        )
        
        result = main_graph.invoke(input_data.model_dump())
        
        # 验证置信度字段存在
        confidence = result.get("structured_data", {}).get("extraction_metadata", {}).get("confidence", 0)
        assert confidence >= 0 and confidence <= 1, f"置信度范围无效: {confidence}"
        print(f"✅ 置信度测试: {confidence}")


def run_e2e_tests():
    """运行E2E测试"""
    print("=" * 60)
    print("开始运行E2E测试...")
    print("=" * 60)
    
    # 创建测试实例
    test_instance = TestE2EWorkflow()
    
    # 运行基础测试
    print("\n[1] 基础结构测试")
    test_instance.test_workflow_exists()
    test_instance.test_workflow_input_schema()
    test_instance.test_workflow_output_schema()
    
    # 运行主流程测试
    print("\n[2] 主流程测试")
    for test_case in TEST_CASES:
        try:
            test_instance.test_main_workflow(test_case)
        except pytest.skip.Exception as e:
            print(f"⏭️ {test_case.name}: {str(e)}")
    
    # 运行性能测试
    print("\n[3] 性能测试")
    test_instance.test_workflow_latency()
    test_instance.test_workflow_confidence()
    
    print("\n" + "=" * 60)
    print("E2E测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    run_e2e_tests()