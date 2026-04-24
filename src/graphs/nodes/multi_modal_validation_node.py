# -*- coding: utf-8 -*-
"""
多模态验证节点（V1.2新增）
使用逻辑推理和一致性检查验证提取的信息
"""

import os
import traceback
from typing import Dict, Any, List, Optional
from datetime import datetime
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context

from graphs.state import (
    MultiModalValidationInput,
    MultiModalValidationOutput
)


def validate_production_date(production_date: str, current_date: str = None) -> Dict[str, Any]:
    """验证生产日期"""
    import re

    if not production_date:
        return {"is_valid": False, "error": "生产日期为空"}

    # 尝试提取日期（支持多种格式）
    date_patterns = [
        r"(\d{4})[年/-](\d{1,2})[月/-](\d{1,2})[日]?",  # YYYY年MM月DD日
        r"(\d{4})(\d{2})(\d{2})",  # YYYYMMDD
        r"(\d{2})[年/-](\d{1,2})[月/-](\d{1,2})[日]?"  # YYMMDD
    ]

    matched = False
    year, month, day = 0, 0, 0

    for pattern in date_patterns:
        match = re.search(pattern, production_date)
        if match:
            matched = True
            groups = match.groups()
            if len(groups) == 3:
                year = int(groups[0])
                month = int(groups[1])
                day = int(groups[2])

                # 处理两位年份
                if year < 100:
                    year += 2000 if year >= 50 else 1900
            break

    if not matched:
        return {"is_valid": False, "error": "生产日期格式无法识别"}

    # 验证日期有效性
    import calendar
    try:
        calendar.monthrange(year, month)  # 检查月份和日期是否有效
    except:
        return {"is_valid": False, "error": f"无效日期: {year}-{month}-{day}"}

    # 检查是否为未来日期
    today = datetime.now()
    prod_date = datetime(year, month, day)

    if prod_date > today:
        return {"is_valid": False, "error": f"生产日期为未来日期: {prod_date}"}

    # 检查是否过早（超过10年）
    if (today - prod_date).days > 3650:
        return {"is_valid": False, "error": f"生产日期过早: {prod_date}"}

    return {"is_valid": True, "normalized_date": f"{year}-{month:02d}-{day:02d}"}


def validate_expiry_date(expiry_date: str, production_date: str = None) -> Dict[str, Any]:
    """验证有效期"""
    import re

    if not expiry_date:
        return {"is_valid": False, "error": "有效期为空"}

    # 提取年份+月份（有效期通常只有年月）
    pattern = r"(\d{4})[年/-]?(\d{1,2})[月]?"
    match = re.search(pattern, expiry_date)

    if not match:
        # 尝试提取完整日期
        date_patterns = [
            r"(\d{4})[年/-](\d{1,2})[月/-](\d{1,2})[日]?",
            r"(\d{4})(\d{2})(\d{2})",
        ]

        for pattern in date_patterns:
            match = re.search(pattern, expiry_date)
            if match:
                break

        if not match:
            return {"is_valid": False, "error": "有效期格式无法识别"}

    groups = match.groups()

    if len(groups) >= 2:
        year = int(groups[0])
        month = int(groups[1])

        # 如果有日
        day = int(groups[2]) if len(groups) >= 3 else 1

        import calendar
        try:
            calendar.monthrange(year, month)
        except:
            return {"is_valid": False, "error": f"无效日期: {year}-{month}"}

        # 检查是否已过期
        today = datetime.now()
        exp_date = datetime(year, month, day)

        if exp_date < today:
            return {
                "is_valid": False,
                "error": f"已过期: {exp_date}",
                "is_expired": True,
                "days_expired": (today - exp_date).days
            }

        # 检查有效期与生产日期的关系
        if production_date:
            prod_result = validate_production_date(production_date)
            if prod_result["is_valid"]:
                prod_date_str = prod_result["normalized_date"]
                prod_date = datetime.strptime(prod_date_str, "%Y-%m-%d")

                # 有效期应该晚于生产日期
                if exp_date <= prod_date:
                    return {"is_valid": False, "error": f"有效期早于生产日期: {exp_date} <= {prod_date}"}

                # 检查有效期长度（通常1-5年）
                days_diff = (exp_date - prod_date).days
                if days_diff < 30:  # 少于30天不合理
                    return {"is_valid": False, "error": f"有效期过短: {days_diff}天"}
                if days_diff > 3650:  # 超过10年不合理
                    return {"is_valid": False, "error": f"有效期过长: {days_diff/365:.1f}年"}

        return {
            "is_valid": True,
            "normalized_date": f"{year}-{month:02d}",
            "days_to_expiry": (exp_date - today).days
        }

    return {"is_valid": False, "error": "有效期格式无法识别"}


def multi_modal_validation_node(
    state: MultiModalValidationInput,
    config: RunnableConfig,
    runtime: Runtime[Context]
) -> MultiModalValidationOutput:
    """
    title: 多模态验证
    desc: 使用逻辑推理和一致性检查验证提取的信息，确保数据准确性和合理性
    integrations: -
    """
    ctx = runtime.context

    print(f"[多模态验证] 开始验证...")
    print(f"[多模态验证] 配置: 逻辑验证={state.enable_logic_validation}, 一致性检查={state.enable_consistency_check}")

    try:
        import re

        start_time = datetime.now()

        is_valid = True
        validation_results = {}
        corrected_info = {}
        validation_errors = []

        # 复制原始信息
        info = state.extracted_info.copy()

        # 1. 验证生产日期
        if state.enable_logic_validation:
            production_date = info.get("production_date", "")
            if production_date:
                print(f"[多模态验证] 验证生产日期: {production_date}")
                result = validate_production_date(production_date)
                validation_results["production_date"] = result

                if not result["is_valid"]:
                    is_valid = False
                    validation_errors.append({
                        "field": "production_date",
                        "error": result["error"],
                        "value": production_date
                    })
                    print(f"[多模态验证] ❌ 生产日期验证失败: {result['error']}")
                else:
                    corrected_info["production_date"] = result["normalized_date"]
                    print(f"[多模态验证] ✅ 生产日期验证通过: {result['normalized_date']}")
            else:
                validation_results["production_date"] = {"is_valid": False, "error": "未提供生产日期"}

        # 2. 验证有效期
        if state.enable_logic_validation:
            expiry_date = info.get("expiry_date", "")
            production_date = info.get("production_date", "")

            if expiry_date:
                print(f"[多模态验证] 验证有效期: {expiry_date}")
                result = validate_expiry_date(expiry_date, production_date)
                validation_results["expiry_date"] = result

                if not result["is_valid"]:
                    is_valid = False
                    validation_errors.append({
                        "field": "expiry_date",
                        "error": result["error"],
                        "value": expiry_date
                    })
                    print(f"[多模态验证] ❌ 有效期验证失败: {result['error']}")
                else:
                    # 检查是否临期（30天内）
                    if "days_to_expiry" in result and result["days_to_expiry"] <= 30:
                        validation_errors.append({
                            "field": "expiry_date",
                            "error": "临期警告",
                            "value": expiry_date,
                            "days_to_expiry": result["days_to_expiry"]
                        })
                        print(f"[多模态验证] ⚠️  临期警告: 剩余{result['days_to_expiry']}天")

                    corrected_info["expiry_date"] = result["normalized_date"]
                    print(f"[多模态验证] ✅ 有效期验证通过: {result['normalized_date']}")
            else:
                validation_results["expiry_date"] = {"is_valid": False, "error": "未提供有效期"}

        # 3. 验证条形码
        if state.enable_logic_validation:
            barcode = info.get("barcode", "")
            if barcode:
                print(f"[多模态验证] 验证条形码: {barcode}")
                # 验证条形码格式（8, 12, 13位数字）
                barcode_clean = re.sub(r"[^0-9]", "", barcode)

                if len(barcode_clean) in [8, 12, 13]:
                    # EAN-13校验位验证
                    if len(barcode_clean) == 13:
                        digits = [int(d) for d in barcode_clean]
                        check_sum = sum(digits[::2]) + sum(d * 3 for d in digits[1:-1:2])
                        check_digit = (10 - (check_sum % 10)) % 10

                        if check_digit != digits[-1]:
                            is_valid = False
                            validation_errors.append({
                                "field": "barcode",
                                "error": f"条形码校验位错误: 计算值={check_digit}, 实际值={digits[-1]}",
                                "value": barcode
                            })
                            validation_results["barcode"] = {"is_valid": False, "error": "校验位错误"}
                            print(f"[多模态验证] ❌ 条形码校验失败")
                        else:
                            corrected_info["barcode"] = barcode_clean
                            validation_results["barcode"] = {"is_valid": True}
                            print(f"[多模态验证] ✅ 条形码验证通过")
                    else:
                        corrected_info["barcode"] = barcode_clean
                        validation_results["barcode"] = {"is_valid": True}
                        print(f"[多模态验证] ✅ 条形码格式正确")
                else:
                    is_valid = False
                    validation_errors.append({
                        "field": "barcode",
                        "error": f"条形码格式错误: 期望8/12/13位数字，实际{len(barcode_clean)}位",
                        "value": barcode
                    })
                    validation_results["barcode"] = {"is_valid": False, "error": "格式错误"}
                    print(f"[多模态验证] ❌ 条形码格式错误")

        # 4. 一致性检查
        if state.enable_consistency_check:
            print(f"[多模态验证] 执行一致性检查...")

            # 检查生产日期和有效期的关系
            if "production_date" in corrected_info and "expiry_date" in corrected_info:
                try:
                    prod_date = datetime.strptime(corrected_info["production_date"], "%Y-%m-%d")

                    # 解析有效期（可能是年月）
                    expiry_str = corrected_info["expiry_date"]
                    if len(expiry_str) == 7:  # YYYY-MM
                        expiry_date = datetime.strptime(expiry_str + "-01", "%Y-%m-%d")
                    else:
                        expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d")

                    if expiry_date < prod_date:
                        is_valid = False
                        validation_errors.append({
                            "field": "consistency",
                            "error": "一致性错误: 有效期早于生产日期",
                            "production_date": corrected_info["production_date"],
                            "expiry_date": corrected_info["expiry_date"]
                        })
                        print(f"[多模态验证] ❌ 一致性检查失败: 有效期早于生产日期")
                    else:
                        print(f"[多模态验证] ✅ 一致性检查通过")
                except:
                    pass

        # 5. 合并修正后的信息
        final_info = info.copy()
        final_info.update(corrected_info)

        processing_time = (datetime.now() - start_time).total_seconds()

        print(f"[多模态验证] 验证完成，耗时: {processing_time:.2f}秒")
        print(f"[多模态验证] 验证结果: {'通过' if is_valid else '失败'}")
        print(f"[多模态验证] 验证错误数: {len(validation_errors)}")

        return MultiModalValidationOutput(
            is_valid=is_valid,
            validation_results=validation_results,
            corrected_info=final_info,
            validation_errors=validation_errors,
            processing_time=processing_time
        )

    except Exception as e:
        print(f"[多模态验证] 处理失败: {e}")
        traceback.print_exc()
        raise
