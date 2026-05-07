"""Pydantic models for DFx tasks and rule packs.

Why pydantic?
- Validates user-supplied YAML/JSON early with helpful error messages.
- Gives the rest of the codebase a single typed surface to consume.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class DFxRuleCheck(BaseModel):
    """Structured rule-check spec consumed by ``rule_checker.py``."""

    model_config = ConfigDict(extra="allow")

    type: Literal[
        "numeric_min",
        "numeric_max",
        "boolean_true",
        "boolean_false",
        "ratio_max",
        "ratio_min",
        "in_set",
    ]
    field: Optional[str] = None
    min: Optional[float] = None
    max: Optional[float] = None
    numerator: Optional[str] = None
    denominator: Optional[str] = None
    allowed: Optional[list[Any]] = None


class DFxRule(BaseModel):
    """A single Design-for-X rule."""

    id: str
    description: str
    check: DFxRuleCheck
    severity: Literal["low", "medium", "high"] = "medium"
    rationale: Optional[str] = None


class RulePack(BaseModel):
    """A named collection of DFx rules."""

    rule_pack: str
    rules: list[DFxRule]

    def by_id(self, rule_id: str) -> DFxRule | None:
        for r in self.rules:
            if r.id == rule_id:
                return r
        return None


class EvaluationCriteria(BaseModel):
    """How a task should be graded."""

    model_config = ConfigDict(extra="allow")

    rule_pack: str
    ground_truth_design: dict[str, Any] = Field(default_factory=dict)
    expected_violations: list[str] = Field(default_factory=list)
    required_fields: list[str] = Field(
        default_factory=lambda: ["summary", "dfx_rule_analysis", "final_recommendation"]
    )
    total_subtasks: int = 1


class DFxTask(BaseModel):
    """A single DFx task / session.

    A scenario is a sequence of these tasks linked by ``scenario_name`` and
    ordered by ``session_id``.
    """

    model_config = ConfigDict(extra="allow")

    task_id: str
    scenario_name: str
    session_id: int
    input_description: str
    design_context: dict[str, Any] = Field(default_factory=dict)
    dfx_rules_path: Optional[str] = None
    constraints: list[str] = Field(default_factory=list)
    expected_output_format: str = "structured_json"
    hidden_dependency_from_previous_sessions: Optional[str] = None
    evaluation_criteria: EvaluationCriteria

    def short_label(self) -> str:
        return f"{self.scenario_name}::session_{self.session_id}::{self.task_id}"
