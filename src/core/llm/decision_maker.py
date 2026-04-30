# -*- coding: utf-8 -*-
"""
LLM决策器
条件触发LLM调用，实现精准提效
"""

import logging
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from core import LLMDecisionMaker as BaseLLMDecisionMaker, LLMTriggerCondition, OCRResult


logger = logging.getLogger(__name__)


@dataclass
class TriggerResult:
    """触发结果"""
    should_trigger: bool
    reason: str
    confidence: float = 0.0
    suggested_action: str = ""


class SmartLLMDecisionMaker(BaseLLMDecisionMaker):
    """
    智能LLM决策器
    基于置信度、冲突、格式等多维度判断是否触发LLM
    """

    # 核心字段及其置信度阈值
    CORE_FIELD_THRESHOLDS = {
        'expiry': 0.85,      # 效期
        'batch': 0.80,       # 批号
        'production_date': 0.85,  # 生产日期
        'brand': 0.75,       # 品牌
        'specification': 0.80,    # 规格
        'manufacturer': 0.75,     # 厂家
    }

    def __init__(self):
        super().__init__()
        # 默认触发条件
        self._init_default_conditions()

    def _init_default_conditions(self):
        """初始化默认触发条件"""
        # 低置信度触发
        self.add_trigger(LLMTriggerCondition(
            condition_type='low_confidence',
            threshold=0.85,
            field_types=['expiry', 'batch', 'production_date']
        ))

        # 格式错误触发
        self.add_trigger(LLMTriggerCondition(
            condition_type='format_error',
            threshold=0.0,
            field_types=['expiry', 'batch', 'production_date']
        ))

    def should_trigger_for_field(
        self,
        field_type: str,
        ocr_confidence: float,
        ocr_text: str,
        raw_image_path: Optional[str] = None
    ) -> TriggerResult:
        """
        判断是否需要触发LLM处理特定字段
        """
        threshold = self.CORE_FIELD_THRESHOLDS.get(field_type, 0.80)

        # 条件1：置信度低于阈值
        if ocr_confidence < threshold:
            return TriggerResult(
                should_trigger=True,
                reason=f"置信度 {ocr_confidence:.2f} < 阈值 {threshold}",
                confidence=ocr_confidence,
                suggested_action=f"llm_correction:{field_type}"
            )

        # 条件2：格式校验失败
        format_error = self._check_format_error(field_type, ocr_text)
        if format_error:
            return TriggerResult(
                should_trigger=True,
                reason=f"格式错误: {format_error}",
                confidence=ocr_confidence,
                suggested_action=f"llm_correction:{field_type}"
            )

        # 条件3：日期逻辑校验失败
        if field_type in ['expiry', 'production_date']:
            logic_error = self._check_date_logic(field_type, ocr_text)
            if logic_error:
                return TriggerResult(
                    should_trigger=True,
                    reason=f"日期逻辑错误: {logic_error}",
                    confidence=ocr_confidence,
                    suggested_action=f"llm_correction:{field_type}"
                )

        # 条件4：关键字段为空
        if not ocr_text or ocr_text.strip() in ['N/A', '', 'null', 'None']:
            return TriggerResult(
                should_trigger=True,
                reason=f"字段为空",
                confidence=0.0,
                suggested_action=f"llm_extraction:{field_type}"
            )

        return TriggerResult(
            should_trigger=False,
            reason=f"置信度达标，格式正确",
            confidence=ocr_confidence,
            suggested_action="use_ocr_result"
        )

    def _check_format_error(self, field_type: str, text: str) -> Optional[str]:
        """格式校验"""
        if not text:
            return None

        text = text.strip()

        # 日期格式校验
        if field_type in ['expiry', 'production_date']:
            date_patterns = [
                r'\d{4}[-/年]\d{1,2}[-/月]\d{1,2}日?',  # 2025/01/15 或 2025年01月15日
                r'\d{4}\d{2}\d{2}',  # 20250115
                r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',  # 01/15/25
            ]
            for pattern in date_patterns:
                if re.search(pattern, text):
                    return None
            return f"日期格式不符合规范: {text}"

        # 批号格式校验
        if field_type == 'batch':
            # 批号通常是字母+数字组合
            if re.match(r'^[A-Za-z0-9]{4,20}$', text):
                return None
            # 允许一些特殊字符
            if re.match(r'^[A-Za-z0-9\-\.]{4,20}$', text):
                return None
            return f"批号格式异常: {text}"

        return None

    def _check_date_logic(self, field_type: str, text: str) -> Optional[str]:
        """日期逻辑校验"""
        if not text:
            return None

        try:
            # 尝试解析日期
            parsed_date = self._parse_date(text)
            if parsed_date is None:
                return None

            now = datetime.now()

            if field_type == 'production_date':
                # 生产日期不能晚于当前时间
                if parsed_date > now:
                    return f"生产日期晚于当前时间"

            if field_type == 'expiry':
                # 效期应该晚于生产日期（这里假设生产日期已知）
                # 效期不能早于某个合理范围（比如当天）
                if parsed_date < now:
                    return f"效期已过期"

            return None

        except Exception:
            return None

    def _parse_date(self, text: str) -> Optional[datetime]:
        """解析日期文本"""
        text = text.strip()

        # 尝试多种格式
        formats = [
            '%Y%m%d',
            '%Y-%m-%d',
            '%Y/%m/%d',
            '%Y年%m月%d日',
            '%y%m%d',
            '%y-%m-%d',
            '%m/%d/%Y',
        ]

        for fmt in formats:
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                continue

        # 尝试提取数字日期
        match = re.search(r'(\d{4})[年/\-]?(\d{1,2})[月/\-]?(\d{1,2})', text)
        if match:
            try:
                return datetime(int(match.group(1)), int(match.group(2)), int(match.group(3)))
            except ValueError:
                pass

        return None


def should_trigger_llm(
    field_type: str,
    ocr_confidence: float,
    ocr_text: str
) -> TriggerResult:
    """
    快速判断函数：是否需要触发LLM
    """
    maker = SmartLLMDecisionMaker()
    return maker.should_trigger_for_field(field_type, ocr_confidence, ocr_text)


def validate_llm_output(
    field_type: str,
    llm_output: Dict[str, Any],
    min_confidence: float = 0.7
) -> Dict[str, Any]:
    """
    校验LLM输出
    确保格式正确、置信度达标
    """
    result = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }

    # 检查置信度
    confidence = llm_output.get('confidence', 1.0)
    if confidence < min_confidence:
        result['is_valid'] = False
        result['errors'].append(f"置信度 {confidence:.2f} < 最低要求 {min_confidence}")

    # 检查必填字段
    value = llm_output.get('value')
    if not value or str(value).strip() in ['N/A', '', 'null', 'None']:
        result['warnings'].append("输出值为空")
        result['is_valid'] = False

    # 检查日期格式
    if field_type in ['expiry', 'production_date']:
        date_value = str(value) if value else ''
        maker = SmartLLMDecisionMaker()
        format_error = maker._check_format_error(field_type, date_value)
        if format_error:
            result['warnings'].append(format_error)

    return result
