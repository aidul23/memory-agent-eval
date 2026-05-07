"""Deterministic mock LLM used for offline tests and CI runs.

The mock parses the latest user message looking for a JSON block describing
the candidate design + rule pack and returns a structured DFx response. This
lets the entire evaluation pipeline be exercised without API access.

It also peeks at retrieved memory hints (delivered in the system prompt) so
that *memory-augmented* mock runs out-perform the *stateless* baseline -
useful for verifying that the pipeline correctly routes memory into prompts.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from .base_llm import BaseLLM, LLMResponse

_DESIGN_BLOCK_RE = re.compile(
    r"<DESIGN_JSON>(?P<json>.*?)</DESIGN_JSON>", re.DOTALL
)
_RULES_BLOCK_RE = re.compile(
    r"<RULES_JSON>(?P<json>.*?)</RULES_JSON>", re.DOTALL
)
_MEMORY_BLOCK_RE = re.compile(
    r"<RETRIEVED_MEMORY>(?P<body>.*?)</RETRIEVED_MEMORY>", re.DOTALL
)


def _check_rule(rule: dict[str, Any], design: dict[str, Any]) -> tuple[str, str]:
    """Pure-Python rule evaluator mirroring ``rule_checker.py``.

    Kept self-contained so the mock has no inter-module dependency cycle.
    Returns (status, explanation).
    """
    spec = rule.get("check", {}) or {}
    kind = spec.get("type")
    field = spec.get("field")
    try:
        if kind == "numeric_min":
            v = float(design.get(field))
            ok = v >= float(spec["min"])
            return ("satisfied" if ok else "violated",
                    f"{field}={v} (min {spec['min']})")
        if kind == "numeric_max":
            v = float(design.get(field))
            ok = v <= float(spec["max"])
            return ("satisfied" if ok else "violated",
                    f"{field}={v} (max {spec['max']})")
        if kind == "boolean_true":
            ok = bool(design.get(field))
            return ("satisfied" if ok else "violated", f"{field}={design.get(field)}")
        if kind == "boolean_false":
            ok = not bool(design.get(field))
            return ("satisfied" if ok else "violated", f"{field}={design.get(field)}")
        if kind == "ratio_max":
            num = float(design.get(spec["numerator"]))
            den = float(design.get(spec["denominator"]))
            ratio = num / den if den else float("inf")
            ok = ratio <= float(spec["max"])
            return ("satisfied" if ok else "violated",
                    f"{spec['numerator']}/{spec['denominator']}={ratio:.2f}")
        if kind == "ratio_min":
            num = float(design.get(spec["numerator"]))
            den = float(design.get(spec["denominator"]))
            ratio = num / den if den else 0.0
            ok = ratio >= float(spec["min"])
            return ("satisfied" if ok else "violated",
                    f"{spec['numerator']}/{spec['denominator']}={ratio:.2f}")
        if kind == "in_set":
            ok = design.get(field) in (spec.get("allowed") or [])
            return ("satisfied" if ok else "violated", f"{field}={design.get(field)}")
    except Exception as exc:  # noqa: BLE001
        return ("uncertain", f"could not evaluate: {exc}")
    return ("uncertain", f"unknown rule type {kind}")


class MockLLM(BaseLLM):
    """Deterministic mock useful for offline integration tests."""

    provider = "mock"

    def generate(self, messages: list[dict[str, str]], temperature: float = 0.0) -> LLMResponse:
        t0 = time.perf_counter()
        joined_input = "\n".join(m.get("content", "") for m in messages)

        design = self._extract_block(joined_input, _DESIGN_BLOCK_RE)
        rules = self._extract_block(joined_input, _RULES_BLOCK_RE) or []
        memory_block = _MEMORY_BLOCK_RE.search(joined_input)
        memory_hits = (memory_block.group("body") if memory_block else "").strip()
        has_memory = bool(memory_hits) and memory_hits.lower() != "none"

        analysis: list[dict[str, Any]] = []
        used_memory: list[dict[str, str]] = []

        # Without memory, the mock is intentionally a bit lazy: it correctly
        # flags only "high"-severity issues. With memory it analyses every
        # rule, so the metrics module can observe a real lift.
        for rule in (rules or []):
            severity = rule.get("severity", "medium")
            if not has_memory and severity != "high":
                analysis.append({
                    "rule_id": rule.get("id"),
                    "status": "uncertain",
                    "explanation": "Insufficient prior context to evaluate.",
                })
                continue
            status, explanation = _check_rule(rule, design or {})
            analysis.append({
                "rule_id": rule.get("id"),
                "status": status,
                "explanation": explanation,
            })

        if has_memory:
            for line in memory_hits.splitlines():
                line = line.strip(" -*")
                if line.startswith("[") and "]" in line:
                    mem_id = line.split("]", 1)[0].strip("[ ")
                    used_memory.append({
                        "memory_id": mem_id,
                        "how_used": "Reused prior reflection to focus the analysis.",
                    })
                    if len(used_memory) >= 3:
                        break

        violated = [a["rule_id"] for a in analysis if a["status"] == "violated"]
        confidence = 0.92 if has_memory else 0.6

        response = {
            "summary": (
                "Memory-augmented review flagged "
                f"{len(violated)} violations." if has_memory
                else "Stateless first-pass review of the candidate design."
            ),
            "decision": "reject" if violated else "approve",
            "dfx_rule_analysis": analysis,
            "used_memory": used_memory,
            "final_recommendation": (
                "Address all listed rule violations and re-submit."
                if violated else
                "Design satisfies all evaluated DFx rules; proceed."
            ),
            "confidence": confidence,
        }

        text = json.dumps(response, indent=2)
        in_tok = max(1, len(joined_input) // 4)
        out_tok = max(1, len(text) // 4)
        return LLMResponse(
            text=text,
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_s=time.perf_counter() - t0,
            estimated_cost_usd=0.0,
            finish_reason="stop",
            raw={"provider": self.provider, "model": self.model},
        )

    @staticmethod
    def _extract_block(text: str, pattern: re.Pattern[str]) -> Any:
        m = pattern.search(text)
        if not m:
            return None
        try:
            return json.loads(m.group("json"))
        except json.JSONDecodeError:
            return None
