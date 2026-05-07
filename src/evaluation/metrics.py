"""Metric helpers used by the evaluator and analysis modules.

Single-interaction metrics live here. Cross-run metrics (improvement slope,
consistency variance, baseline lift) are computed from the aggregated
DataFrame in ``src/analysis/aggregate_results.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass
class MetricsCalculator:
    """Stateless namespace for metric helpers."""

    @staticmethod
    def rule_compliance(
        check_results: list[Any],
    ) -> tuple[float, list[str]]:
        """Return (compliance_score in [0,1], list of violated rule ids)."""
        if not check_results:
            return 1.0, []
        violated = [r.rule_id for r in check_results if r.status == "violated"]
        evaluable = [r for r in check_results if r.status in {"satisfied", "violated"}]
        if not evaluable:
            return 0.0, violated
        satisfied = [r for r in evaluable if r.status == "satisfied"]
        return len(satisfied) / len(evaluable), violated

    @staticmethod
    def progress_score(correct_subtasks: int, total_subtasks: int) -> float:
        if total_subtasks <= 0:
            return 0.0
        return max(0.0, min(1.0, correct_subtasks / total_subtasks))

    @staticmethod
    def task_success(rule_compliance_score: float, threshold: float = 0.999) -> bool:
        """A task is *successful* iff every evaluable rule is satisfied.

        We use a near-1.0 threshold to allow for floating-point slack.
        """
        return rule_compliance_score >= threshold

    @staticmethod
    def memory_utility(
        retrieved: list[Any],
        agent_used: list[dict[str, Any]],
        ground_truth_violations: Iterable[str],
    ) -> dict[str, float]:
        """Quantify how useful retrieved memory was for this interaction.

        - retrieval_relevance: avg score of retrieved items (already in [0,1]).
        - retrieval_correctness: fraction of retrieved items whose metadata
          mentions a rule from the ground-truth violation set or shares the
          scenario name.
        - usage_quality: fraction of retrieved items that the agent reported
          actually using (`used_memory[].memory_id`).
        """
        gt = set(ground_truth_violations or [])
        if not retrieved:
            return {
                "retrieval_relevance": 0.0,
                "retrieval_correctness": 0.0,
                "usage_quality": 0.0,
            }
        used_ids = {u.get("memory_id") for u in (agent_used or []) if u}
        rel = sum(getattr(r, "score", 0.0) for r in retrieved) / len(retrieved)

        def _is_correct(item: Any) -> bool:
            blob = (getattr(item, "content", "") or "")
            md = getattr(item, "metadata", {}) or {}
            return any(rid in blob for rid in gt) or any(rid in str(md) for rid in gt)

        correct = sum(1 for r in retrieved if _is_correct(r)) / len(retrieved)
        usage = sum(1 for r in retrieved if r.id in used_ids) / len(retrieved)
        return {
            "retrieval_relevance": round(rel, 4),
            "retrieval_correctness": round(correct, 4),
            "usage_quality": round(usage, 4),
        }
