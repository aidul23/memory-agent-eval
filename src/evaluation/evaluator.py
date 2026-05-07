"""End-to-end task evaluator.

Glues ``RuleChecker`` + ``MetricsCalculator`` + ``FeedbackGenerator`` so the
agent loop only needs to call ``Evaluator.evaluate(task, response, ...)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..memory.base_memory import MemoryItem
from ..tasks.dfx_task import DFxTask
from ..tasks.task_loader import load_rule_pack
from .feedback_generator import FeedbackGenerator
from .metrics import MetricsCalculator
from .rule_checker import RuleCheckResult, RuleChecker


@dataclass
class EvaluationResult:
    task_success: bool
    rule_compliance_score: float
    violated_rules: list[str]
    correct_subtasks: int
    total_subtasks: int
    rule_results: list[RuleCheckResult] = field(default_factory=list)
    feedback: dict[str, Any] = field(default_factory=dict)
    memory_usage_quality: dict[str, float] = field(default_factory=dict)
    progress_score: float = 0.0


class Evaluator:
    """Computes structured evaluation + feedback for one task interaction."""

    def __init__(self, default_rules_path: str | None = None) -> None:
        self.default_rules_path = default_rules_path

    def evaluate(
        self,
        task: DFxTask,
        agent_response: dict[str, Any],
        retrieved_memory: list[MemoryItem] | None = None,
    ) -> EvaluationResult:
        rules_path = task.dfx_rules_path or self.default_rules_path
        if not rules_path:
            raise ValueError(f"Task {task.task_id} has no dfx_rules_path and no default was provided.")
        rule_pack = load_rule_pack(rules_path)
        checker = RuleChecker(rule_pack)

        gt_design = task.evaluation_criteria.ground_truth_design
        check_results = checker.check_all(gt_design)

        compliance, violated_rules = MetricsCalculator.rule_compliance(check_results)
        success = MetricsCalculator.task_success(compliance)

        # Subtask = correctly identifying each rule's status.
        # We compare the agent's reported analysis against the ground truth.
        agent_analysis = agent_response.get("dfx_rule_analysis") or []
        agent_status: dict[str, str] = {
            (a.get("rule_id") or ""): (a.get("status") or "uncertain")
            for a in agent_analysis if isinstance(a, dict)
        }
        correct_subtasks = 0
        for r in check_results:
            if agent_status.get(r.rule_id) == r.status:
                correct_subtasks += 1
        total_subtasks = max(
            len(check_results),
            int(task.evaluation_criteria.total_subtasks),
        )

        memory_quality = MetricsCalculator.memory_utility(
            retrieved=retrieved_memory or [],
            agent_used=agent_response.get("used_memory") or [],
            ground_truth_violations=violated_rules,
        )

        feedback = FeedbackGenerator(rule_pack).build(
            task_success=success,
            rule_compliance_score=compliance,
            check_results=check_results,
            correct_subtasks=correct_subtasks,
            total_subtasks=total_subtasks,
            memory_usage_quality=memory_quality,
        )

        return EvaluationResult(
            task_success=success,
            rule_compliance_score=compliance,
            violated_rules=violated_rules,
            correct_subtasks=correct_subtasks,
            total_subtasks=total_subtasks,
            rule_results=check_results,
            feedback=feedback,
            memory_usage_quality=memory_quality,
            progress_score=MetricsCalculator.progress_score(correct_subtasks, total_subtasks),
        )
