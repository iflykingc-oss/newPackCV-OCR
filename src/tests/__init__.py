# -*- coding: utf-8 -*-
"""
单元测试套件
"""

import os
import sys
import json
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from io import BytesIO

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.cv.preprocessor import ImagePreprocessor
from src.core.ocr.ocr_scheduler import OCRScheduler
from src.core.rule_engine.validator import ExpiryValidator
from src.core.rule_engine.alert import AlertManager
from src.core.llm.decision_maker import LLMEvidenceEvaluator


class TestImagePreprocessor:
    """图像预处理测试"""

    @pytest.fixture
    def preprocessor(self):
        return ImagePreprocessor()

    @pytest.fixture
    def sample_image(self):
        """生成测试图片"""
        img = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        return img

    def test_denoise(self, preprocessor, sample_image):
        """测试去噪"""
        result = preprocessor.denoise(sample_image)
        assert result is not None
        assert result.shape == sample_image.shape

    def test_clahe_enhance(self, preprocessor, sample_image):
        """测试CLAHE增强"""
        result = preprocessor.clahe_enhance(sample_image)
        assert result is not None
        assert result.shape == sample_image.shape

    def test_sharpen(self, preprocessor, sample_image):
        """测试锐化"""
        result = preprocessor.sharpen(sample_image)
        assert result is not None
        assert result.shape == sample_image.shape

    def test_full_pipeline(self, preprocessor, sample_image):
        """测试完整预处理流程"""
        result = preprocessor.process(sample_image, enable_all=True)
        assert result is not None
        assert "enhanced_image" in result


class TestOCRScheduler:
    """OCR调度器测试"""

    @pytest.fixture
    def scheduler(self):
        return OCRScheduler()

    @pytest.fixture
    def sample_image_bytes(self):
        """生成测试图片字节"""
        from PIL import Image
        img = Image.new('RGB', (200, 100), color='white')
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        return buffer.getvalue()

    @patch('src.core.ocr.ocr_scheduler.TesseractOCR')
    def test_tesseract_recognize(self, mock_tesseract, scheduler, sample_image_bytes):
        """测试Tesseract识别"""
        mock_tesseract.return_value.recognize.return_value = MagicMock(
            raw_text="Test Text",
            confidence=0.9,
            is_success=True
        )

        result = scheduler.recognize(sample_image_bytes, engine="tesseract")
        assert result is not None
        assert result.raw_text == "Test Text"

    def test_engine_fallback(self, scheduler, sample_image_bytes):
        """测试引擎降级"""
        # 当所有引擎都失败时，应该返回默认结果
        result = scheduler.recognize(sample_image_bytes)
        assert result is not None


class TestExpiryValidator:
    """效期校验器测试"""

    @pytest.fixture
    def validator(self):
        return ExpiryValidator()

    def test_valid_date_format(self, validator):
        """测试有效日期格式"""
        result = validator.validate_format("2025-12-31")
        assert result["is_valid"] is True

    def test_invalid_date_format(self, validator):
        """测试无效日期格式"""
        result = validator.validate_format("2025/13/01")
        assert result["is_valid"] is False

    def test_future_date(self, validator):
        """测试未来日期"""
        result = validator.validate_date("2099-12-31")
        assert result["is_valid"] is False

    def test_shelf_life_calculation(self, validator):
        """测试保质期计算"""
        data = {
            "production_date": "2025-01-01",
            "shelf_life": "365"
        }
        result = validator.validate(data)
        assert "days_until_expiry" in result


class TestAlertManager:
    """告警管理器测试"""

    @pytest.fixture
    def alert_manager(self):
        return AlertManager()

    def test_create_alert(self, alert_manager):
        """测试创建告警"""
        alert = alert_manager.create_alert(
            alert_type="expired",
            alert_level="critical",
            message="商品已过期",
            product_name="测试商品"
        )
        assert alert is not None
        assert alert.alert_type == "expired"

    def test_deduplicate(self, alert_manager):
        """测试告警去重"""
        # 创建相同告警
        alert1 = alert_manager.create_alert(
            alert_type="expired",
            product_name="测试商品",
            product_date="2025-01-01"
        )
        alert2 = alert_manager.create_alert(
            alert_type="expired",
            product_name="测试商品",
            product_date="2025-01-01"
        )

        # 应该去重
        is_duplicate = alert_manager.is_duplicate(alert2)
        assert is_duplicate is True

    def test_statistics(self, alert_manager):
        """测试统计功能"""
        # 创建多条告警
        alert_manager.create_alert(alert_type="expired", alert_level="critical")
        alert_manager.create_alert(alert_type="expired", alert_level="warning")
        alert_manager.create_alert(alert_type="approaching", alert_level="info")

        stats = alert_manager.get_statistics()
        assert stats["total"] == 3
        assert stats["by_level"]["critical"] == 1
        assert stats["by_level"]["warning"] == 1


class TestLLMEvidenceEvaluator:
    """LLM证据评估器测试"""

    @pytest.fixture
    def evaluator(self):
        return LLMEvidenceEvaluator()

    def test_should_trigger_llm_low_confidence(self, evaluator):
        """测试低置信度触发LLM"""
        result = evaluator.should_trigger_llm(
            raw_text="Test",
            confidence=0.5,  # 低置信度
            structured_data={}
        )
        assert result is True

    def test_should_trigger_llm_high_confidence(self, evaluator):
        """测试高置信度不触发LLM"""
        result = evaluator.should_trigger_llm(
            raw_text="Test",
            confidence=0.95,  # 高置信度
            structured_data={"product_name": "Test Product"}
        )
        assert result is False

    @patch('src.core.llm.decision_maker.get_llm_client')
    def test_correct_text_with_llm(self, mock_llm, evaluator):
        """测试LLM纠错"""
        mock_client = MagicMock()
        mock_client.chat.return_value = {
            "corrected_text": "Corrected Text",
            "corrections": []
        }
        mock_llm.return_value = mock_client

        result = evaluator.correct_text_with_llm(
            raw_text="Tset",
            image_context={"url": "test.jpg"}
        )
        assert result is not None


# ============ 集成测试 ============

class TestOCRWorkflow:
    """OCR完整工作流测试"""

    @pytest.fixture
    def workflow(self):
        from src.graphs.graph import main_graph
        return main_graph

    def test_simple_ocr_pipeline(self, workflow):
        """测试简单OCR流程"""
        from src.graphs.state import GraphInput
        from utils.file.file import File

        # 创建测试输入
        test_input = GraphInput(
            package_image=File(
                url="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
                file_type="image"
            ),
            ocr_engine_type="builtin"
        )

        # 执行工作流
        # result = workflow.invoke(test_input)
        # assert result is not None
        pytest.skip("需要mock LLM调用")


# 运行测试
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
