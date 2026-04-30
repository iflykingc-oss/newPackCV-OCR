# -*- coding: utf-8 -*-
"""
规则引擎实现
提供日期校验、批号校验、告警规则
"""

import re
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

from core import RuleEngine as BaseRuleEngine, FieldValidationResult


logger = logging.getLogger(__name__)


@dataclass
class ValidationRule:
    """验证规则"""
    name: str
    field_type: str
    pattern: Optional[str] = None
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    custom_validator: Optional[callable] = None
    error_message: str = "验证失败"


class ExpiryValidator(BaseRuleEngine):
    """
    效期规则引擎
    支持：
    - 日期格式校验
    - 逻辑校验（生产日期≤当前日期≤有效期）
    - 临期告警
    """

    # 常见日期格式
    DATE_PATTERNS = [
        (r'\d{4}-\d{2}-\d{2}', '%Y-%m-%d'),      # 2025-01-15
        (r'\d{4}/\d{2}/\d{2}', '%Y/%m/%d'),      # 2025/01/15
        (r'\d{4}\d{2}\d{2}', '%Y%m%d'),          # 20250115
        (r'\d{4}年\d{1,2}月\d{1,2}日', '%Y年%m月%d日'),  # 2025年1月15日
        (r'\d{1,2}/\d{1,2}/\d{4}', '%m/%d/%Y'),  # 01/15/2025
        (r'\d{1,2}/\d{1,2}/\d{2}', '%m/%d/%y'),  # 01/15/25
    ]

    # 保质期范围（月）
    SHELF_LIFE_RANGES = {
        '酒类': 120,       # 10年
        '调味品': 36,      # 3年
        '罐头': 24,        # 2年
        '饮料': 12,        # 1年
        '乳制品': 6,       # 6个月
        '生鲜': 3,         # 3个月
        '默认': 24,        # 默认2年
    }

    def __init__(self, current_date: Optional[datetime] = None):
        self.current_date = current_date or datetime.now()
        self.rules: List[ValidationRule] = []
        self._init_default_rules()

    def _init_default_rules(self):
        """初始化默认规则"""
        # 生产日期规则
        self.rules.append(ValidationRule(
            name='production_date_format',
            field_type='production_date',
            pattern=r'\d{4}[-/年]\d{1,2}[-/月]\d{1,2}',
            error_message='日期格式不正确'
        ))

        # 效期规则
        self.rules.append(ValidationRule(
            name='expiry_format',
            field_type='expiry',
            pattern=r'\d{4}[-/年]\d{1,2}[-/月]\d{1,2}',
            error_message='日期格式不正确'
        ))

        # 批号规则
        self.rules.append(ValidationRule(
            name='batch_format',
            field_type='batch',
            pattern=r'^[A-Za-z0-9\-\.]{4,20}$',
            error_message='批号格式不正确'
        ))

    def validate_expiry(
        self,
        production_date: str,
        expiry_date: str
    ) -> FieldValidationResult:
        """
        效期校验
        检查日期格式和逻辑关系
        """
        result = FieldValidationResult(
            field_name='expiry',
            value=expiry_date,
            is_valid=True,
            confidence=1.0
        )

        # 解析日期
        prod_date = self._parse_date(production_date)
        exp_date = self._parse_date(expiry_date)

        # 格式校验
        if prod_date is None:
            result.is_valid = False
            result.error_message = f"生产日期格式无法识别: {production_date}"
            result.confidence = 0.0
            return result

        if exp_date is None:
            result.is_valid = False
            result.error_message = f"效期格式无法识别: {expiry_date}"
            result.confidence = 0.0
            return result

        # 逻辑校验
        # 1. 生产日期不能晚于当前日期
        if prod_date > self.current_date:
            result.is_valid = False
            result.error_message = f"生产日期 {production_date} 晚于当前日期"
            result.confidence = 0.5
            return result

        # 2. 效期不能早于生产日期
        if exp_date < prod_date:
            result.is_valid = False
            result.error_message = f"效期 {expiry_date} 早于生产日期 {production_date}"
            result.confidence = 0.5
            return result

        # 3. 效期合理性校验（不能超过合理范围）
        delta = exp_date - prod_date
        if delta.days > 365 * 10:  # 超过10年不合理
            result.is_valid = False
            result.error_message = f"保质期超过10年，可能识别错误"
            result.confidence = 0.7
            return result

        # 4. 计算临期天数
        days_until_expiry = (exp_date - self.current_date).days

        if days_until_expiry < 0:
            result.is_valid = False
            result.error_message = f"商品已过期 {-days_until_expiry} 天"
            result.confidence = 1.0
        elif days_until_expiry <= 30:
            result.is_valid = True
            result.error_message = f"商品临期，剩余 {days_until_expiry} 天"
            result.confidence = 0.9
        else:
            result.is_valid = True
            result.error_message = None
            result.confidence = 1.0

        return result

    def validate_batch(self, batch_number: str) -> FieldValidationResult:
        """
        批号校验
        检查格式和逻辑
        """
        result = FieldValidationResult(
            field_name='batch',
            value=batch_number,
            is_valid=True,
            confidence=1.0
        )

        if not batch_number or batch_number.strip() in ['', 'N/A', 'null']:
            result.is_valid = False
            result.error_message = "批号为空"
            result.confidence = 0.0
            return result

        # 格式校验
        if not re.match(r'^[A-Za-z0-9\-\.]{4,20}$', batch_number):
            result.is_valid = False
            result.error_message = f"批号格式异常: {batch_number}"
            result.confidence = 0.5
            return result

        # 尝试从批号中提取日期信息
        date_in_batch = self._extract_date_from_batch(batch_number)
        if date_in_batch:
            # 验证批号日期与生产日期的一致性
            result.confidence = 0.9

        return result

    def check_conflict(
        self,
        ocr_result: Any,
        llm_result: Dict
    ) -> bool:
        """
        冲突检测
        检查OCR和LLM结果是否冲突
        """
        if not ocr_result or not llm_result:
            return False

        ocr_value = str(ocr_result).strip()
        llm_value = str(llm_result.get('value', '')).strip()

        if not ocr_value or not llm_value:
            return False

        # 完全相同，不冲突
        if ocr_value == llm_value:
            return False

        # 计算编辑距离
        distance = self._levenshtein_distance(ocr_value, llm_value)
        max_len = max(len(ocr_value), len(llm_value))

        # 编辑距离超过50%认为是冲突
        if max_len > 0 and distance / max_len > 0.5:
            return True

        return False

    def _parse_date(self, text: str) -> Optional[datetime]:
        """解析日期文本"""
        if not text:
            return None

        text = text.strip()

        for pattern, fmt in self.DATE_PATTERNS:
            match = re.search(pattern, text)
            if match:
                try:
                    return datetime.strptime(match.group(), fmt)
                except ValueError:
                    continue

        # 尝试直接解析
        try:
            return datetime.fromisoformat(text.replace('/', '-'))
        except ValueError:
            pass

        return None

    def _extract_date_from_batch(self, batch: str) -> Optional[datetime]:
        """从批号中提取日期"""
        # 常见格式：YYYYMMDD 或 YYMMDD
        patterns = [
            (r'(\d{4})(\d{2})(\d{2})', '%Y%m%d'),
            (r'(\d{2})(\d{2})(\d{2})', '%y%m%d'),
        ]

        for pattern, fmt in patterns:
            match = re.search(pattern, batch)
            if match:
                try:
                    return datetime.strptime(''.join(match.groups()), fmt)
                except ValueError:
                    continue

        return None

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """计算编辑距离"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]


def create_rule_engine(current_date: Optional[datetime] = None) -> ExpiryValidator:
    """规则引擎工厂函数"""
    return ExpiryValidator(current_date)
