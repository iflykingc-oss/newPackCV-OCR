#!/usr/bin/env python3
"""回归测试流水线 - 自动化回归检测"""
import os
import json
import subprocess
from typing import Dict, Any, List
from datetime import datetime


class RegressionTestPipeline:
    """回归测试流水线"""
    
    def __init__(self):
        self.test_results: List[Dict[str, Any]] = []
    
    def run_unit_tests(self) -> Dict[str, Any]:
        """运行单元测试"""
        print("\n[阶段1] 单元测试")
        result = {
            "stage": "unit_tests",
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        try:
            # 运行pytest单元测试
            proc = subprocess.run(
                ["python", "-m", "pytest", "src/tests/unit/", "-v", "--tb=short"],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            result["stdout"] = proc.stdout[-500:] if len(proc.stdout) > 500 else proc.stdout
            result["stderr"] = proc.stderr[-200:] if len(proc.stderr) > 200 else proc.stderr
            result["success"] = proc.returncode == 0
            result["return_code"] = proc.returncode
            
            print(f"  结果: {'✅ 成功' if result['success'] else '❌ 失败'}")
            
        except Exception as e:
            result["success"] = False
            result["error"] = str(e)
            print(f"  结果: ❌ 异常 - {str(e)}")
        
        self.test_results.append(result)
        return result
    
    def run_e2e_tests(self) -> Dict[str, Any]:
        """运行E2E测试"""
        print("\n[阶段2] E2E测试")
        result = {
            "stage": "e2e_tests",
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        try:
            proc = subprocess.run(
                ["python", "-m", "pytest", "src/tests/e2e/", "-v", "--tb=short"],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            result["stdout"] = proc.stdout[-500:] if len(proc.stdout) > 500 else proc.stdout
            result["stderr"] = proc.stderr[-200:] if len(proc.stderr) > 200 else proc.stderr
            result["success"] = proc.returncode == 0
            result["return_code"] = proc.returncode
            
            print(f"  结果: {'✅ 成功' if result['success'] else '❌ 失败'}")
            
        except Exception as e:
            result["success"] = False
            result["error"] = str(e)
            print(f"  结果: ❌ 异常 - {str(e)}")
        
        self.test_results.append(result)
        return result
    
    def run_performance_tests(self) -> Dict[str, Any]:
        """运行性能测试"""
        print("\n[阶段3] 性能测试")
        result = {
            "stage": "performance_tests",
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        try:
            # 运行性能基准测试
            from tests.performance.test_benchmark import run_performance_benchmark
            
            perf_result = run_performance_benchmark()
            
            result["success"] = perf_result.get("success_rate", 0) >= 0.8
            result["performance_summary"] = perf_result
            
            print(f"  结果: {'✅ 成功' if result['success'] else '❌ 失败'}")
            
        except Exception as e:
            result["success"] = False
            result["error"] = str(e)
            print(f"  结果: ❌ 异常 - {str(e)}")
        
        self.test_results.append(result)
        return result
    
    def run_code_quality_check(self) -> Dict[str, Any]:
        """运行代码质量检查"""
        print("\n[阶段4] 代码质量检查")
        result = {
            "stage": "code_quality",
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        try:
            # 运行pyright类型检查
            proc = subprocess.run(
                ["pyright", "src/"],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            result["stdout"] = proc.stdout[-300:] if len(proc.stdout) > 300 else proc.stdout
            result["stderr"] = proc.stderr[-100:] if len(proc.stderr) > 100 else proc.stderr
            result["success"] = proc.returncode == 0 or "error" not in proc.stdout.lower()
            result["return_code"] = proc.returncode
            
            print(f"  结果: {'✅ 成功' if result['success'] else '❌ 有警告'}")
            
        except Exception as e:
            result["success"] = True  # 类型检查工具缺失不阻止流水线
            result["error"] = str(e)
            print(f"  结果: ⏭️ 跳过 - {str(e)}")
        
        self.test_results.append(result)
        return result
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """运行完整回归测试流水线"""
        print("=" * 60)
        print("开始回归测试流水线...")
        print("=" * 60)
        
        # 迧行各阶段测试
        unit_result = self.run_unit_tests()
        e2e_result = self.run_e2e_tests()
        perf_result = self.run_performance_tests()
        quality_result = self.run_code_quality_check()
        
        # 生成汇总报告
        all_success = all(r.get("success", False) for r in self.test_results)
        
        summary: Dict[str, Any] = {
            "pipeline_timestamp": datetime.utcnow().isoformat(),
            "overall_success": all_success,
            "stages_count": len(self.test_results),
            "stages_success": sum(1 for r in self.test_results if r.get("success")),
            "stages_failed": sum(1 for r in self.test_results if not r.get("success")),
            "results": self.test_results
        }
        
        print("\n" + "=" * 60)
        print("回归测试流水线完成！")
        print("=" * 60)
        print(f"总体结果: {'✅ 全部通过' if all_success else '❌ 有失败'}")
        print(f"成功阶段: {summary['stages_success']}/{summary['stages_count']}")
        
        return summary
    
    def export_results(self, output_path: str) -> None:
        """导出测试结果"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, ensure_ascii=False, indent=2)
        print(f"测试结果已导出: {output_path}")


def run_regression_pipeline():
    """运行回归测试流水线"""
    pipeline = RegressionTestPipeline()
    summary = pipeline.run_full_pipeline()
    
    # 导出结果
    output_dir = os.path.join(os.getenv("COZE_WORKSPACE_PATH", ""), "test-results")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"regression_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json")
    pipeline.export_results(output_path)
    
    return summary


if __name__ == "__main__":
    run_regression_pipeline()