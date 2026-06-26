#!/usr/bin/env python3
"""性能基准测试 - 自动化性能测试"""
import os
import json
import time
import statistics
from typing import Dict, Any, List
from datetime import datetime


class PerformanceBenchmark:
    """性能基准测试类"""
    
    def __init__(self):
        self.results: List[Dict[str, Any]] = []
        self.test_iterations: int = 5
    
    def run_single_test(self, test_name: str, input_data: dict) -> Dict[str, Any]:
        """运行单次性能测试"""
        from graphs.graph import main_graph
        
        start_time = time.time()
        result = main_graph.invoke(input_data)
        elapsed_time = time.time() - start_time
        
        return {
            "test_name": test_name,
            "elapsed_time_seconds": elapsed_time,
            "timestamp": datetime.utcnow().isoformat(),
            "success": result is not None,
            "result_keys": list(result.keys()) if result else []
        }
    
    def run_benchmark_suite(self) -> Dict[str, Any]:
        """运行完整性能基准测试套件"""
        print("=" * 60)
        print("开始性能基准测试...")
        print("=" * 60)
        
        test_inputs = [
            ("basic_image", {"input_file": {"url": "https://picsum.photos/seed/bench1/200/300.jpg", "file_type": "image"}}),
            ("with_question", {"input_file": {"url": "https://picsum.photos/seed/bench2/200/300.jpg", "file_type": "image"}, "user_question": "描述图片"}),
            ("large_image", {"input_file": {"url": "https://picsum.photos/seed/bench3/1920/1080.jpg", "file_type": "image"}}),
        ]
        
        all_results: List[Dict[str, Any]] = []
        
        for test_name, input_data in test_inputs:
            print(f"\n[测试] {test_name}")
            test_results: List[float] = []
            
            for i in range(self.test_iterations):
                try:
                    result = self.run_single_test(f"{test_name}_run{i+1}", input_data)
                    test_results.append(result["elapsed_time_seconds"])
                    print(f"  运行 {i+1}: {result['elapsed_time_seconds']:.2f}秒")
                    all_results.append(result)
                except Exception as e:
                    print(f"  运行 {i+1}: 失败 - {str(e)}")
                    all_results.append({
                        "test_name": f"{test_name}_run{i+1}",
                        "error": str(e),
                        "success": False
                    })
            
            if test_results:
                avg_time = statistics.mean(test_results)
                std_dev = statistics.stdev(test_results) if len(test_results) > 1 else 0
                print(f"  平均: {avg_time:.2f}秒, 标准差: {std_dev:.2f}秒")
        
        # 生成汇总报告
        summary = self._generate_summary(all_results)
        
        print("\n" + "=" * 60)
        print("性能基准测试完成！")
        print("=" * 60)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        
        return summary
    
    def _generate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成性能汇总报告"""
        successful_results = [r for r in results if r.get("success")]
        failed_results = [r for r in results if not r.get("success")]
        
        times = [r["elapsed_time_seconds"] for r in successful_results if "elapsed_time_seconds" in r]
        
        summary: Dict[str, Any] = {
            "test_timestamp": datetime.utcnow().isoformat(),
            "total_runs": len(results),
            "successful_runs": len(successful_results),
            "failed_runs": len(failed_results),
            "success_rate": len(successful_results) / len(results) if results else 0,
        }
        
        if times:
            summary["performance_stats"] = {
                "avg_time_seconds": round(statistics.mean(times), 3),
                "min_time_seconds": round(min(times), 3),
                "max_time_seconds": round(max(times), 3),
                "std_dev_seconds": round(statistics.stdev(times), 3) if len(times) > 1 else 0,
                "p95_time_seconds": round(sorted(times)[int(len(times) * 0.95)] if len(times) >= 20 else max(times), 3),
            }
        
        return summary


def run_performance_benchmark():
    """运行性能基准测试"""
    benchmark = PerformanceBenchmark()
    return benchmark.run_benchmark_suite()


if __name__ == "__main__":
    run_performance_benchmark()