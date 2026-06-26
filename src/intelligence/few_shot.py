"""Few-shot 示例库 - 提升 LLM 提取准确率

通过场景化示例注入 Prompt，准确率可提升 5-15%。
"""
import time
import uuid
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


class FewShotExample(BaseModel):
    """单个 Few-shot 示例"""
    example_id: str = Field(default_factory=lambda: f"ex-{uuid.uuid4().hex[:12]}")
    scenario: str = Field(..., description="场景标识")
    input_summary: str = Field(..., description="输入摘要或描述")
    output: str = Field(..., description="期望输出")
    score: float = Field(default=0.0, description="用户评分 0-1")
    use_count: int = Field(default=0, description="被引用次数")
    created_at: float = Field(default_factory=time.time)
    tags: List[str] = Field(default_factory=list, description="标签")

    def to_prompt(self) -> str:
        """转换为 Prompt 片段"""
        score_str = f" [score={self.score:.2f}]" if self.score > 0 else ""
        tags_str = f" tags={','.join(self.tags)}" if self.tags else ""
        return (
            f"### Example (id={self.example_id}){score_str}{tags_str}\n"
            f"Input: {self.input_summary}\n"
            f"Output: {self.output}\n"
        )


class FewShotManager:
    """Few-shot 库管理器（内存版）"""

    def __init__(self, max_per_scenario: int = 50):
        self._examples: Dict[str, FewShotExample] = {}
        self.max_per_scenario = max_per_scenario

    def add_example(
        self,
        scenario: str,
        input_summary: str,
        output: str,
        score: float = 0.0,
        tags: Optional[List[str]] = None,
    ) -> FewShotExample:
        """添加示例"""
        # 限制每场景数量
        existing = [e for e in self._examples.values() if e.scenario == scenario]
        if len(existing) >= self.max_per_scenario:
            # 移除评分最低的
            existing.sort(key=lambda e: e.score)
            self._examples.pop(existing[0].example_id, None)

        ex = FewShotExample(
            scenario=scenario,
            input_summary=input_summary,
            output=output,
            score=score,
            tags=tags or [],
        )
        self._examples[ex.example_id] = ex
        return ex

    def get_examples(
        self, scenario: str, top_k: int = 3, min_score: float = 0.0
    ) -> List[FewShotExample]:
        """获取场景的 top-k 示例（按评分排序）"""
        candidates = [
            e for e in self._examples.values()
            if e.scenario == scenario and e.score >= min_score
        ]
        candidates.sort(key=lambda e: (e.score, e.use_count), reverse=True)
        return candidates[:top_k]

    def record_use(self, example_ids: List[str]) -> None:
        """记录使用次数"""
        for eid in example_ids:
            ex = self._examples.get(eid)
            if ex:
                ex.use_count += 1

    def rate_example(self, example_id: str, score: float) -> bool:
        """用户评分"""
        ex = self._examples.get(example_id)
        if not ex:
            return False
        ex.score = max(0.0, min(1.0, score))
        return True

    def delete_example(self, example_id: str) -> bool:
        return self._examples.pop(example_id, None) is not None

    def build_prompt(self, scenario: str, top_k: int = 3, min_score: float = 0.0) -> str:
        """构建 Few-shot Prompt 片段"""
        examples = self.get_examples(scenario, top_k, min_score)
        if not examples:
            return ""
        return "\n".join(ex.to_prompt() for ex in examples)

    def list_scenarios(self) -> List[Dict[str, Any]]:
        """列出所有场景及示例数"""
        counter: Dict[str, int] = {}
        for e in self._examples.values():
            counter[e.scenario] = counter.get(e.scenario, 0) + 1
        return [{"scenario": s, "count": c} for s, c in counter.items()]

    def stats(self) -> Dict[str, Any]:
        return {
            "total_examples": len(self._examples),
            "scenarios": len(self.list_scenarios()),
            "max_per_scenario": self.max_per_scenario,
        }
