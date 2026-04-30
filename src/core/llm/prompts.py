# -*- coding: utf-8 -*-
"""
LLM提示词模板
标准化、结构化的提示词设计
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class PromptTemplate:
    """提示词模板"""
    system_prompt: str
    user_prompt_template: str

    def render(self, **kwargs) -> tuple:
        """
        渲染提示词
        Returns: (system_prompt, user_prompt)
        """
        import re

        # 渲染用户提示词
        user_prompt = self.user_prompt_template
        for key, value in kwargs.items():
            placeholder = f"{{{key}}}"
            user_prompt = user_prompt.replace(placeholder, str(value))

        return self.system_prompt, user_prompt


# ==================== 纠错提示词 ====================

CORRECTION_SYSTEM_PROMPT = """你是一位专业的包装OCR识别纠错专家。你的任务是：
1. 基于商品包装图像和原始OCR识别结果，进行智能纠错
2. 纠错原则：保持原意、最小修改、可信度优先
3. 输出严格的JSON格式，包含纠错后的值和置信度

关键约束：
- 只修改明显错误（如OCR误识别、缺字、多字）
- 日期格式必须符合标准（如2025-01-15）
- 批号必须保持字母数字组合
- 如果无法确定，输出原始OCR结果

输出格式：
{
  "corrected_value": "纠错后的值",
  "confidence": 0.0-1.0,
  "corrections": ["具体修改说明"],
  "reason": "纠错理由"
}"""

CORRECTION_USER_TEMPLATE = """## 任务：商品包装OCR纠错

## 原始OCR识别结果
- 字段类型：{field_type}
- OCR识别值：{ocr_value}
- OCR置信度：{ocr_confidence}

## 图像信息
- 图片URL：{image_url}

## 纠错要求
1. 仔细核对图像中的实际文字
2. 修正OCR识别的错误
3. 保持原始信息的准确性
4. 格式标准化（如日期转为YYYY-MM-DD）

请输出纠错结果JSON："""

CORRECTION_PROMPT = PromptTemplate(
    system_prompt=CORRECTION_SYSTEM_PROMPT,
    user_prompt_template=CORRECTION_USER_TEMPLATE
)


# ==================== 结构化提取提示词 ====================

EXTRACTION_SYSTEM_PROMPT = """你是一位专业的商品信息结构化提取专家。你的任务是：
1. 从商品包装图像中提取关键信息
2. 识别字段类型并标准化输出
3. 处理模糊、倾斜、复杂排版

支持的字段类型：
- brand: 品牌名称
- product_name: 产品名称
- specification: 规格（如500ml, 100g）
- production_date: 生产日期
- expiry_date: 保质期/到期日期
- batch_number: 批号
- manufacturer: 生产商
- ingredients: 配料/成分
- storage_method: 存储方式
- net_weight: 净含量

关键约束：
- 日期必须标准化为 YYYY-MM-DD 格式
- 规格需包含单位和数值
- 批号保持原始格式
- 所有字段必须输出，即使是空值

输出格式：
{
  "fields": {
    "brand": {"value": "...", "confidence": 0.95},
    "product_name": {"value": "...", "confidence": 0.90},
    ...
  },
  "overall_confidence": 0.92,
  "warnings": ["无法识别的区域说明"]
}"""

EXTRACTION_USER_TEMPLATE = """## 任务：商品包装信息结构化提取

## 图像信息
- 图片URL：{image_url}
- 图片类型：{image_type}（单品/多品）

## 需要提取的字段
{required_fields}

## 提取要求
1. 仔细识别图像中的所有文字
2. 将文字分类到对应的字段
3. 对每个字段给出置信度评分
4. 无法识别的字段标记为null

请输出结构化结果JSON："""

EXTRACTION_PROMPT = PromptTemplate(
    system_prompt=EXTRACTION_SYSTEM_PROMPT,
    user_prompt_template=EXTRACTION_USER_TEMPLATE
)


# ==================== 验证提示词 ====================

VALIDATION_SYSTEM_PROMPT = """你是一位商品合规性校验专家。你的任务是：
1. 校验商品信息的完整性和准确性
2. 检查日期逻辑（生产日期≤当前日期≤有效期）
3. 识别潜在的合规问题

校验规则：
- 生产日期不能晚于当前日期
- 有效期不能早于生产日期
- 批号格式必须符合规范
- 品牌名称必须完整

输出格式：
{
  "is_valid": true/false,
  "errors": ["错误描述"],
  "warnings": ["警告描述"],
  "compliance_score": 0.0-1.0
}"""

VALIDATION_USER_TEMPLATE = """## 任务：商品信息校验

## 待校验信息
{product_info}

## 当前日期
{current_date}

## 校验要求
1. 检查日期逻辑
2. 验证格式规范
3. 评估合规性

请输出校验结果JSON："""

VALIDATION_PROMPT = PromptTemplate(
    system_prompt=VALIDATION_SYSTEM_PROMPT,
    user_prompt_template=VALIDATION_USER_TEMPLATE
)


# ==================== 冲突仲裁提示词 ====================

CONFLICT_SYSTEM_PROMPT = """你是一位OCR结果仲裁专家。当多个OCR引擎给出不同结果时，你需要：
1. 结合图像上下文做出最终判断
2. 给出判断理由
3. 评估最终结果的置信度

判断原则：
- 优先选择更清晰的识别结果
- 参考字体大小和位置
- 考虑常见的OCR错误模式

输出格式：
{
  "final_value": "最终确定的值",
  "confidence": 0.0-1.0,
  "reason": "判断理由",
  "alternative_values": ["其他可能的值及概率"]
}"""

CONFLICT_USER_TEMPLATE = """## 任务：OCR结果冲突仲裁

## 图像信息
- 图片URL：{image_url}

## 多引擎识别结果
{ocr_results}

## 仲裁要求
1. 基于图像判断正确结果
2. 解释选择理由
3. 给出最终值和置信度

请输出仲裁结果JSON："""

CONFLICT_PROMPT = PromptTemplate(
    system_prompt=CONFLICT_SYSTEM_PROMPT,
    user_prompt_template=CONFLICT_USER_TEMPLATE
)


# ==================== 辅助函数 ====================

def get_correction_prompt(
    field_type: str,
    ocr_value: str,
    ocr_confidence: float,
    image_url: str
) -> tuple:
    """获取纠错提示词"""
    return CORRECTION_PROMPT.render(
        field_type=field_type,
        ocr_value=ocr_value,
        ocr_confidence=ocr_confidence,
        image_url=image_url
    )


def get_extraction_prompt(
    image_url: str,
    image_type: str = "单品",
    required_fields: Optional[List[str]] = None
) -> tuple:
    """获取结构化提取提示词"""
    if required_fields is None:
        required_fields = [
            "brand", "product_name", "specification",
            "production_date", "expiry_date", "batch_number"
        ]

    fields_str = "\n".join([f"- {f}" for f in required_fields])

    return EXTRACTION_PROMPT.render(
        image_url=image_url,
        image_type=image_type,
        required_fields=fields_str
    )


def get_validation_prompt(
    product_info: Dict[str, Any],
    current_date: str
) -> tuple:
    """获取校验提示词"""
    import json
    info_str = json.dumps(product_info, ensure_ascii=False, indent=2)

    return VALIDATION_PROMPT.render(
        product_info=info_str,
        current_date=current_date
    )
