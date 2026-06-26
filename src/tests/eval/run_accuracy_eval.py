#!/usr/bin/env python3
"""
准确率评测脚本 - 跑评测集统计准确率
用于回归测试和性能基准
"""
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

# 加载评测集
EVAL_SET_PATH = Path(__file__).parent.parent / "assets" / "eval" / "accuracy_eval_set.json"


def load_eval_set() -> Dict[str, Any]:
    with open(EVAL_SET_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def evaluate_case(case: Dict, result: Dict) -> Dict[str, Any]:
    """评估单个测试用例"""
    expected = case.get("expected_fields", {})
    actual = result.get("structured_data", {})

    correct = 0
    total = len(expected)

    for field, constraint in expected.items():
        if field in actual and actual[field]:
            # 简单验证：非空
            if "非空" in constraint or "non-empty" in constraint:
                correct += 1
            elif "YYYY-MM-DD" in constraint:
                if isinstance(actual[field], str) and len(actual[field]) >= 8:
                    correct += 1
            elif "浮点数" in constraint or "float" in constraint:
                if isinstance(actual[field], (int, float)) and actual[field] > 0:
                    correct += 1
            elif "字符串" in constraint or "string" in constraint:
                if isinstance(actual[field], str) and len(actual[field]) > 0:
                    correct += 1
            else:
                correct += 1  # 简化处理

    accuracy = correct / total if total > 0 else 0.0
    return {
        "case_id": case.get("case_id"),
        "accuracy": accuracy,
        "passed": accuracy >= case.get("accuracy_threshold", 0.9),
        "correct_fields": correct,
        "total_fields": total
    }


def run_evaluation(test_results: List[Dict] = None) -> Dict[str, Any]:
    """
    运行评测
    test_results: 实际测试结果列表，每个元素包含 case_id 和 structured_data
    """
    eval_set = load_eval_set()
    results = []

    # 如果没有传入实际结果，使用模拟数据演示
    if test_results is None:
        print("⚠️  未提供实际测试结果，仅生成评测框架报告")
        return {
            "total_cases": eval_set["total_cases"],
            "framework_version": eval_set["version"],
            "status": "framework_ready"
        }

    for result in test_results:
        # 查找对应的测试用例
        case_id = result.get("case_id")
        case = None
        for scenario, cfg in eval_set["scenarios"].items():
            for c in cfg.get("test_cases", []):
                if c.get("case_id") == case_id:
                    case = c
                    break
            if case:
                break

        if case:
            eval_result = evaluate_case(case, result)
            results.append(eval_result)

    # 统计
    total_cases = len(results)
    passed_cases = sum(1 for r in results if r.get("passed"))
    avg_accuracy = sum(r.get("accuracy", 0) for r in results) / total_cases if total_cases > 0 else 0

    return {
        "total_cases": total_cases,
        "passed_cases": passed_cases,
        "failed_cases": total_cases - passed_cases,
        "pass_rate": passed_cases / total_cases if total_cases > 0 else 0,
        "avg_accuracy": avg_accuracy,
        "results": results
    }


if __name__ == "__main__":
    print("=" * 60)
    print("PackCV-OCR 准确率评测")
    print("=" * 60)

    # 演示模式
    report = run_evaluation()

    if report.get("status") == "framework_ready":
        print(f"\n✅ 评测框架就绪")
        print(f"  - 评测集版本: {report['framework_version']}")
        print(f"  - 总用例数: {report['total_cases']}")
        print(f"\n📋 使用方法:")
        print(f"  1. 准备实际测试结果")
        print(f"  2. 调用 run_evaluation(test_results=results)")
        print(f"  3. 查看报告")
    else:
        print(f"\n📊 评测报告:")
        print(f"  - 总用例: {report['total_cases']}")
        print(f"  - 通过: {report['passed_cases']}")
        print(f"  - 失败: {report['failed_cases']}")
        print(f"  - 通过率: {report['pass_rate']*100:.1f}%")
        print(f"  - 平均准确率: {report['avg_accuracy']*100:.1f}%")
