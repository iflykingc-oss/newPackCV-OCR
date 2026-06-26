"""A/B 测试框架 - 模型/策略/场景对比

通过哈希分桶保证一致性，使用 Z 检验判定显著度。
"""
import hashlib
import math
import time
import uuid
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class Variant(BaseModel):
    """变体"""
    name: str
    weight: int = Field(ge=1, le=100, description="权重 1-100")
    config: Dict[str, Any] = Field(default_factory=dict)


class ExperimentConfig(BaseModel):
    """实验配置"""
    experiment_id: str = Field(default_factory=lambda: f"exp-{uuid.uuid4().hex[:8]}")
    name: str
    description: str = ""
    variants: List[Variant]
    created_at: float = Field(default_factory=time.time)
    enabled: bool = True

    def validate(self) -> None:
        if not self.variants:
            raise ValueError("至少 1 个变体")
        if sum(v.weight for v in self.variants) <= 0:
            raise ValueError("变体权重总和必须 > 0")


def get_z_score(p1: float, n1: int, p2: float, n2: int) -> float:
    """两比例 Z 检验"""
    if n1 == 0 or n2 == 0:
        return 0.0
    p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
    if p_pool in (0, 1):
        return 0.0
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0
    return (p2 - p1) / se


class ABTestFramework:
    """A/B 测试框架"""

    def __init__(self, experiment: ExperimentConfig):
        experiment.validate()
        self.experiment = experiment
        self._assignments: Dict[str, str] = {}  # user_id -> variant
        self._results: Dict[str, Dict[str, int]] = {}  # variant -> {success, total}

    def assign(self, user_id: str) -> str:
        """分桶（基于 user_id 哈希，保证一致）"""
        if not self.experiment.enabled:
            return self.experiment.variants[0].name
        if user_id in self._assignments:
            return self._assignments[user_id]

        # 哈希分桶
        h = hashlib.md5(
            f"{self.experiment.experiment_id}:{user_id}".encode()
        ).hexdigest()
        bucket = int(h, 16) % 100
        cumulative = 0
        total_weight = sum(v.weight for v in self.experiment.variants)
        for v in self.experiment.variants:
            cumulative += v.weight * 100 // total_weight
            if bucket < cumulative:
                self._assignments[user_id] = v.name
                return v.name
        return self.experiment.variants[-1].name

    def get_variant_config(self, user_id: str) -> Dict[str, Any]:
        """获取分桶对应的配置"""
        variant_name = self.assign(user_id)
        for v in self.experiment.variants:
            if v.name == variant_name:
                return v.config
        return {}

    def record(self, user_id: str, success: bool) -> None:
        """记录结果"""
        variant = self.assign(user_id)
        if variant not in self._results:
            self._results[variant] = {"success": 0, "total": 0}
        self._results[variant]["total"] += 1
        if success:
            self._results[variant]["success"] += 1

    def report(self) -> Dict[str, Any]:
        """生成报告"""
        report = {"variants": [], "winner": None, "significant": False}
        for v in self.experiment.variants:
            r = self._results.get(v.name, {"success": 0, "total": 0})
            rate = r["success"] / r["total"] if r["total"] > 0 else 0
            report["variants"].append({
                "name": v.name,
                "weight": v.weight,
                "success": r["success"],
                "total": r["total"],
                "success_rate": round(rate, 4),
            })

        # 显著度对比
        if len(report["variants"]) >= 2 and report["variants"][0]["total"] >= 30:
            v0 = report["variants"][0]
            best = max(report["variants"][1:], key=lambda v: v["success_rate"])
            z = get_z_score(
                v0["success_rate"], v0["total"],
                best["success_rate"], best["total"],
            )
            report["z_score"] = round(z, 4)
            report["p_value"] = round(2 * (1 - _normal_cdf(abs(z))), 4)
            report["significant"] = abs(z) > 1.96  # 95% 置信
            if report["significant"]:
                report["winner"] = best["name"]
        return report


def _normal_cdf(z: float) -> float:
    """标准正态分布 CDF（近似）"""
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))
