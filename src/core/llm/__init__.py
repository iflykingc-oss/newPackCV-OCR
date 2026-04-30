# -*- coding: utf-8 -*-
"""
LLM融合决策层模块
提供LLM条件触发、纠错、结构化提取能力
"""

from core.llm.decision_maker import (
    LLMDecisionMaker,
    LLMTriggerCondition,
    should_trigger_llm,
    validate_llm_output
)
from core.llm.prompts import (
    PromptTemplate,
    CORRECTION_PROMPT,
    EXTRACTION_PROMPT,
    VALIDATION_PROMPT
)

__all__ = [
    'LLMDecisionMaker',
    'LLMTriggerCondition',
    'should_trigger_llm',
    'validate_llm_output',
    'PromptTemplate',
    'CORRECTION_PROMPT',
    'EXTRACTION_PROMPT',
    'VALIDATION_PROMPT'
]
