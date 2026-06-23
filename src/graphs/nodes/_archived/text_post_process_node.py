# -*- coding: utf-8 -*-
"""
文本后处理节点（V1.1新增）
支持半全角转换、文本纠错、格式规范化等功能
"""

import os
import traceback
from typing import Dict, Any, List, Optional
from datetime import datetime
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context

from graphs.state import (
    TextPostProcessInput,
    TextPostProcessOutput
)


class TextPostProcessor:
    """文本后处理器"""

    def __init__(self):
        self.corrections = []

    def process(self, text: str, options: Dict[str, bool]) -> str:
        """处理文本"""
        processed = text
        self.corrections = []

        # 1. 半全角转换
        if options.get("enable_full_half_convert", False):
            processed = self.convert_full_half(processed)
            print(f"[文本后处理] 执行半全角转换")

        # 2. 文本纠错
        if options.get("enable_spell_correct", False):
            processed = self.correct_spelling(processed)
            print(f"[文本后处理] 执行文本纠错")

        # 3. 格式规范化
        if options.get("enable_format_normalize", False):
            processed = self.normalize_format(processed)
            print(f"[文本后处理] 执行格式规范化")

        # 4. 清理多余空格
        if options.get("enable_cleanup_whitespace", False):
            processed = self.cleanup_whitespace(processed)
            print(f"[文本后处理] 清理多余空格")

        return processed

    def convert_full_half(self, text: str) -> str:
        """半全角转换"""
        # 全角数字转半角
        full_to_half_num = {
            "０": "0", "１": "1", "２": "2", "３": "3", "４": "4",
            "５": "5", "６": "6", "７": "7", "８": "8", "９": "9"
        }

        # 全角字母转半角
        full_to_half_upper = {
            "Ａ": "A", "Ｂ": "B", "Ｃ": "C", "Ｄ": "D", "Ｅ": "E",
            "Ｆ": "F", "Ｇ": "G", "Ｈ": "H", "Ｉ": "I", "Ｊ": "J",
            "Ｋ": "K", "Ｌ": "L", "Ｍ": "M", "Ｎ": "N", "Ｏ": "O",
            "Ｐ": "P", "Ｑ": "Q", "Ｒ": "R", "Ｓ": "S", "Ｔ": "T",
            "Ｕ": "U", "Ｖ": "V", "Ｗ": "W", "Ｘ": "X", "Ｙ": "Y", "Ｚ": "Z"
        }

        full_to_half_lower = {
            "ａ": "a", "ｂ": "b", "ｃ": "c", "ｄ": "d", "ｅ": "e",
            "ｆ": "f", "ｇ": "g", "ｈ": "h", "ｉ": "i", "ｊ": "j",
            "ｋ": "k", "ｌ": "l", "ｍ": "m", "ｎ": "n", "ｏ": "o",
            "ｐ": "p", "ｑ": "q", "ｒ": "r", "ｓ": "s", "ｔ": "t",
            "ｕ": "u", "ｖ": "v", "ｗ": "w", "ｘ": "x", "ｙ": "y", "ｚ": "z"
        }

        # 全角标点转半角
        full_to_half_punct = {
            "，": ",", "。": ".", "！": "!", "？": "?",
            "：": ":", "；": ";", "（": "(", "）": ")",
            "【": "[", "】": "]", "｛": "{", "｝": "}",
            "－": "-", "／": "/", "＼": "\\", "｜": "|"
        }

        # 合并所有映射
        full_to_half = {}
        full_to_half.update(full_to_half_num)
        full_to_half.update(full_to_half_upper)
        full_to_half.update(full_to_half_lower)
        full_to_half.update(full_to_half_punct)

        # 执行转换
        result = []
        for char in text:
            if char in full_to_half:
                original = char
                converted = full_to_half[char]
                result.append(converted)
                if original != converted:
                    self.corrections.append({
                        "type": "full_half_convert",
                        "original": original,
                        "converted": converted
                    })
            else:
                result.append(char)

        return "".join(result)

    def correct_spelling(self, text: str) -> str:
        """文本纠错（简化版本，实际可以使用pycorrector等库）"""
        # 这里使用简单的规则纠错
        # 实际应用中可以使用pycorrector、哈工大LTP等工具

        # 常见错误映射
        common_errors = {
            "生产曰期": "生产日期",
            "效期": "有效期",
            "保质期": "有效期",
            "产品名": "产品名称",
            "规恪": "规格",
            "净含量": "净含量"
        }

        result = text
        for error, correct in common_errors.items():
            if error in result:
                result = result.replace(error, correct)
                self.corrections.append({
                    "type": "spelling_correct",
                    "original": error,
                    "converted": correct
                })

        return result

    def normalize_format(self, text: str) -> str:
        """格式规范化"""
        # 1. 日期格式规范化
        import re

        # 统一日期格式为 YYYY-MM-DD
        date_patterns = [
            r"(\d{4})年(\d{1,2})月(\d{1,2})日",
            r"(\d{4})/(\d{1,2})/(\d{1,2})",
            r"(\d{4})\.(\d{1,2})\.(\d{1,2})",
            r"(\d{4})(\d{2})(\d{2})"
        ]

        for pattern in date_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                original = match.group(0)
                year, month, day = match.groups()
                normalized = f"{year}-{month.zfill(2)}-{day.zfill(2)}"

                if original != normalized:
                    text = text.replace(original, normalized)
                    self.corrections.append({
                        "type": "date_normalize",
                        "original": original,
                        "converted": normalized
                    })

        # 2. 数字格式规范化（千分位）
        # 匹配大数字（大于4位）
        number_pattern = r"\b(\d{4,})\b"
        matches = re.finditer(number_pattern, text)
        for match in matches:
            original = match.group(1)
            # 添加千分位分隔符
            normalized = "{:,}".format(int(original))

            if original != normalized:
                text = text.replace(original, normalized)
                self.corrections.append({
                    "type": "number_normalize",
                    "original": original,
                    "converted": normalized
                })

        return text

    def cleanup_whitespace(self, text: str) -> str:
        """清理多余空格"""
        import re

        # 1. 去除行首行尾空格
        lines = [line.strip() for line in text.split("\n")]

        # 2. 合并连续空行（最多保留一个空行）
        cleaned_lines = []
        prev_empty = False

        for line in lines:
            if line:
                cleaned_lines.append(line)
                prev_empty = False
            elif not prev_empty:
                cleaned_lines.append("")
                prev_empty = True

        # 3. 去除最后一行空行
        while cleaned_lines and not cleaned_lines[-1]:
            cleaned_lines.pop()

        # 4. 合并行内多余空格（保留英文单词间的一个空格）
        final_lines = []
        for line in cleaned_lines:
            # 压缩多个空格为一个
            line = re.sub(r" +", " ", line)
            final_lines.append(line)

        return "\n".join(final_lines)


def text_post_process_node(state: TextPostProcessInput, config: RunnableConfig, runtime: Runtime[Context]) -> TextPostProcessOutput:
    """
    title: 文本后处理
    desc: 支持半全角转换、文本纠错、格式规范化等功能，提升文本质量
    integrations: -
    """
    ctx = runtime.context

    print(f"[文本后处理] 开始处理文本...")
    print(f"[文本后处理] 配置: 半全角转换={state.enable_full_half_convert}, 文本纠错={state.enable_spell_correct}, 格式规范化={state.enable_format_normalize}, 清理空格={state.enable_cleanup_whitespace}")

    try:
        start_time = datetime.now()

        processor = TextPostProcessor()

        # 构建处理选项
        options = {
            "enable_full_half_convert": state.enable_full_half_convert,
            "enable_spell_correct": state.enable_spell_correct,
            "enable_format_normalize": state.enable_format_normalize,
            "enable_cleanup_whitespace": state.enable_cleanup_whitespace
        }

        # 处理文本
        processed_text = processor.process(state.text, options)

        # 构建处理步骤
        processing_steps = []
        if state.enable_full_half_convert:
            processing_steps.append("半全角转换")
        if state.enable_spell_correct:
            processing_steps.append("文本纠错")
        if state.enable_format_normalize:
            processing_steps.append("格式规范化")
        if state.enable_cleanup_whitespace:
            processing_steps.append("清理空格")

        processing_time = (datetime.now() - start_time).total_seconds()

        print(f"[文本后处理] 处理完成，耗时: {processing_time:.2f}秒")
        print(f"[文本后处理] 纠错数量: {len(processor.corrections)}")
        if processor.corrections:
            print(f"[文本后处理] 纠正示例: {processor.corrections[:3]}")

        return TextPostProcessOutput(
            processed_text=processed_text,
            corrections=processor.corrections,
            correction_count=len(processor.corrections),
            processing_steps=processing_steps,
            processing_time=processing_time
        )

    except Exception as e:
        print(f"[文本后处理] 处理失败: {e}")
        traceback.print_exc()
        raise
