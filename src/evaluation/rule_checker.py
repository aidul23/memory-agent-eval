"""Deterministic rule checker.

Given a rule pack and a *ground-truth* design, returns which rules are
satisfied / violated. This is the platform's source of truth for rule
compliance - the agent's self-reported analysis is compared against it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..tasks.dfx_task import DFxRule, RulePack


@dataclass
class RuleCheckResult:
    rule_id: str
    status: str          # 'satisfied' | 'violated' | 'uncertain'
    explanation: str
    severity: str = "medium"

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "status": self.status,
            "explanation": self.explanation,
            "severity": self.severity,
        }


class RuleChecker:
    """Evaluates a single design against a rule pack."""

    def __init__(self, rule_pack: RulePack) -> None:
        self.rule_pack = rule_pack

    def check_all(self, design: dict[str, Any]) -> list[RuleCheckResult]:
        return [self.check(rule, design) for rule in self.rule_pack.rules]

    def check(self, rule: DFxRule, design: dict[str, Any]) -> RuleCheckResult:
        spec = rule.check
        kind = spec.type
        try:
            if kind == "numeric_min":
                v = float(design.get(spec.field))  # type: ignore[arg-type]
                ok = v >= float(spec.min)  # type: ignore[arg-type]
                expl = f"{spec.field}={v} (min {spec.min})"
            elif kind == "numeric_max":
                v = float(design.get(spec.field))  # type: ignore[arg-type]
                ok = v <= float(spec.max)  # type: ignore[arg-type]
                expl = f"{spec.field}={v} (max {spec.max})"
            elif kind == "boolean_true":
                v = bool(design.get(spec.field))
                ok = v
                expl = f"{spec.field}={v}"
            elif kind == "boolean_false":
                v = bool(design.get(spec.field))
                ok = not v
                expl = f"{spec.field}={v}"
            elif kind == "ratio_max":
                num = float(design.get(spec.numerator))  # type: ignore[arg-type]
                den = float(design.get(spec.denominator))  # type: ignore[arg-type]
                ratio = num / den if den else float("inf")
                ok = ratio <= float(spec.max)  # type: ignore[arg-type]
                expl = f"{spec.numerator}/{spec.denominator}={ratio:.2f} (max {spec.max})"
            elif kind == "ratio_min":
                num = float(design.get(spec.numerator))  # type: ignore[arg-type]
                den = float(design.get(spec.denominator))  # type: ignore[arg-type]
                ratio = num / den if den else 0.0
                ok = ratio >= float(spec.min)  # type: ignore[arg-type]
                expl = f"{spec.numerator}/{spec.denominator}={ratio:.2f} (min {spec.min})"
            elif kind == "in_set":
                v = design.get(spec.field)
                ok = v in (spec.allowed or [])
                expl = f"{spec.field}={v} (allowed {spec.allowed})"
            else:
                return RuleCheckResult(
                    rule_id=rule.id,
                    status="uncertain",
                    explanation=f"Unknown rule type {kind!r}",
                    severity=rule.severity,
                )
        except (TypeError, ValueError, KeyError) as exc:
            return RuleCheckResult(
                rule_id=rule.id,
                status="uncertain",
                explanation=f"Could not evaluate rule {rule.id}: {exc}",
                severity=rule.severity,
            )

        return RuleCheckResult(
            rule_id=rule.id,
            status="satisfied" if ok else "violated",
            explanation=expl,
            severity=rule.severity,
        )
