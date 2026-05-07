"""Memory-augmented DFx agent.

Pipeline:
1. Retrieve top-k memories for the current task.
2. Build a structured prompt that includes:
     - role definition
     - DFx rules
     - task context
     - retrieved memory
     - prior feedback (if any)
     - required output format
3. Call the LLM at temperature 0.0.
4. Parse the JSON response (with a tolerant extractor).
5. Return an ``AgentResult`` with everything for the runner to log.

Note: this class does NOT call evaluator/feedback/memory.update - that is
done by the experiment runner so the memory is updated AFTER evaluation
with the canonical (deterministic) feedback envelope.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..llms.base_llm import BaseLLM
from ..memory.base_memory import BaseMemory, MemoryItem
from ..tasks.dfx_task import DFxTask
from ..tasks.task_loader import load_rule_pack
from ..utils import get_logger
from .base_agent import AgentResult, BaseAgent

logger = get_logger(__name__)

_SYSTEM_PROMPT = """You are an expert Design-for-X (DFx) reviewer.

You will be given:
1. A candidate engineering design (DESIGN_JSON).
2. A rule pack to apply (RULES_JSON).
3. Optional retrieved memory from prior interactions (RETRIEVED_MEMORY).
4. Optional prior feedback (PRIOR_FEEDBACK).

Your job:
- Evaluate every rule against the design.
- For each rule output exactly one of: "satisfied", "violated", "uncertain".
- If you used any retrieved memory, list the corresponding memory_id and how it influenced your reasoning.
- Reply with a SINGLE JSON object, no prose, no markdown fences.
- Required keys: summary, decision, dfx_rule_analysis, used_memory, final_recommendation, confidence.

Output schema:
{
  "summary": "...",
  "decision": "...",
  "dfx_rule_analysis": [{"rule_id": "...", "status": "satisfied|violated|uncertain", "explanation": "..."}],
  "used_memory": [{"memory_id": "...", "how_used": "..."}],
  "final_recommendation": "...",
  "confidence": 0.0
}
"""


class MemoryAgent(BaseAgent):
    """LLM-driven agent with a pluggable ``BaseMemory``."""

    def __init__(
        self,
        llm: BaseLLM,
        memory: BaseMemory,
        temperature: float = 0.0,
    ) -> None:
        self.llm = llm
        self.memory = memory
        self.temperature = temperature

    def run(self, task: DFxTask, prior_feedback: dict[str, Any] | None = None) -> AgentResult:
        retrieved = self.memory.retrieve(
            query=task.input_description,
            context={
                "scenario_name": task.scenario_name,
                "session_id": task.session_id,
                "design_context": task.design_context,
            },
        )

        messages = self._build_messages(task, retrieved, prior_feedback)
        llm_resp = self.llm.generate(messages, temperature=self.temperature)
        parsed = self._parse_response(llm_resp.text)

        return AgentResult(
            response=parsed,
            raw_text=llm_resp.text,
            prompt_messages=messages,
            retrieved_memory=retrieved,
            llm_response=llm_resp,
        )

    # ---- Prompt construction -----------------------------------------

    def _build_messages(
        self,
        task: DFxTask,
        retrieved: list[MemoryItem],
        prior_feedback: dict[str, Any] | None,
    ) -> list[dict[str, str]]:
        rule_pack_path = task.dfx_rules_path or ""
        rules_payload: list[dict[str, Any]] = []
        if rule_pack_path:
            try:
                rp = load_rule_pack(rule_pack_path)
                rules_payload = [
                    {"id": r.id, "description": r.description,
                     "severity": r.severity, "check": r.check.model_dump(exclude_none=True)}
                    for r in rp.rules
                ]
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not load rule pack %s: %s", rule_pack_path, exc)

        retrieved_block = self._format_memory(retrieved)
        prior_feedback_block = (
            json.dumps(prior_feedback, indent=2) if prior_feedback else "None"
        )

        user_blocks = [
            f"TASK_ID: {task.task_id}",
            f"SCENARIO: {task.scenario_name}  SESSION: {task.session_id}",
            f"INPUT_DESCRIPTION: {task.input_description}",
            f"CONSTRAINTS: {json.dumps(task.constraints)}",
            f"<DESIGN_JSON>{json.dumps(task.design_context, indent=2)}</DESIGN_JSON>",
            f"<RULES_JSON>{json.dumps(rules_payload, indent=2)}</RULES_JSON>",
            f"<RETRIEVED_MEMORY>\n{retrieved_block}\n</RETRIEVED_MEMORY>",
            f"<PRIOR_FEEDBACK>\n{prior_feedback_block}\n</PRIOR_FEEDBACK>",
            "Reply with the JSON object only.",
        ]

        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": "\n\n".join(user_blocks)},
        ]

    @staticmethod
    def _format_memory(items: list[MemoryItem]) -> str:
        if not items:
            return "None"
        lines = []
        for it in items:
            lines.append(f"[{it.id}] (score={it.score:.2f}, type={it.type}) {it.content}")
        return "\n".join(lines)

    # ---- Response parsing --------------------------------------------

    _JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
    _BARE_OBJ_RE = re.compile(r"(\{.*\})", re.DOTALL)

    def _parse_response(self, text: str) -> dict[str, Any]:
        """Tolerant JSON extractor.

        LLMs sometimes wrap output in markdown fences or add prose. We try
        a few extraction strategies and return a best-effort dict; if all
        fail we wrap the raw text in a stub object so downstream code keeps
        working (the evaluator will mark fields as missing).
        """
        if not text:
            return self._stub("Empty response from LLM.")

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        m = self._JSON_FENCE_RE.search(text)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass

        m = self._BARE_OBJ_RE.search(text)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass

        return self._stub(text)

    @staticmethod
    def _stub(raw: str) -> dict[str, Any]:
        return {
            "summary": "Could not parse structured response.",
            "decision": "uncertain",
            "dfx_rule_analysis": [],
            "used_memory": [],
            "final_recommendation": "",
            "confidence": 0.0,
            "_parse_error": True,
            "_raw": raw[:2000],
        }
