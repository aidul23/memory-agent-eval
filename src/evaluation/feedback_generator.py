"""Convert raw evaluation results into the structured feedback envelope.

Output schema (see README -> "Feedback format"):

    {
      "task_success": bool,
      "rule_compliance_score": float,
      "violated_rules": [str, ...],
      "correct_subtasks": int,
      "total_subtasks": int,
      "feedback_summary": str,
      "improvement_suggestions": [str, ...],
      "memory_usage_quality": {
        "retrieval_relevance": float,
        "retrieval_correctness": float,
        "usage_quality": float
      }
    }
"""

from __future__ import annotations

from typing import Any

from ..tasks.dfx_task import RulePack
from .rule_checker import RuleCheckResult


class FeedbackGenerator:
    """Builds the structured feedback envelope shown to the agent next round."""

    def __init__(self, rule_pack: RulePack | None = None) -> None:
        self.rule_pack = rule_pack

    def build(
        self,
        *,
        task_success: bool,
        rule_compliance_score: float,
        check_results: list[RuleCheckResult],
        correct_subtasks: int,
        total_subtasks: int,
        memory_usage_quality: dict[str, float],
    ) -> dict[str, Any]:
        violated = [r for r in check_results if r.status == "violated"]
        violated_ids = [r.rule_id for r in violated]
        suggestions = [self._suggestion(r) for r in violated]

        if task_success:
            summary = (
                f"All {total_subtasks} subtasks satisfied; "
                f"{rule_compliance_score:.0%} rule compliance."
            )
        elif violated:
            summary = (
                f"{len(violated)} rule(s) violated: "
                f"{', '.join(violated_ids)}. "
                f"Rule compliance {rule_compliance_score:.0%}."
            )
        else:
            summary = (
                "No explicit violations but task did not pass success threshold; "
                "review ambiguous rules."
            )

        return {
            "task_success": task_success,
            "rule_compliance_score": round(rule_compliance_score, 4),
            "violated_rules": violated_ids,
            "correct_subtasks": correct_subtasks,
            "total_subtasks": total_subtasks,
            "feedback_summary": summary,
            "improvement_suggestions": suggestions,
            "memory_usage_quality": memory_usage_quality,
        }

    def _suggestion(self, result: RuleCheckResult) -> str:
        """Return a human-readable improvement suggestion for one violation."""
        rule = self.rule_pack.by_id(result.rule_id) if self.rule_pack else None
        rationale = rule.rationale if rule else None
        base = f"Address {result.rule_id}: {result.explanation}."
        if rationale:
            return f"{base} Why it matters: {rationale}"
        return base
