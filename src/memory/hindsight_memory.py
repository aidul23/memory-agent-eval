"""HindsightMemory: feedback-driven reflections.

After every interaction the agent stores a short *reflection*:
- which rule(s) were violated,
- what went wrong,
- what to do next time.

On retrieval, the memory returns the top-k reflections most relevant to the
current task description. Because reflections are short, even a simple
token-overlap retrieval works well in practice.
"""

from __future__ import annotations

from typing import Any

from ._similarity import jaccard
from .base_memory import BaseMemory, MemoryItem


class HindsightMemory(BaseMemory):
    name = "hindsight"

    def __init__(
        self,
        max_reflections: int = 50,
        top_k: int = 5,
        use_llm_summarizer: bool = False,
        **_: Any,
    ) -> None:
        self.max_reflections = max_reflections
        self.top_k = top_k
        self.use_llm_summarizer = use_llm_summarizer  # reserved for future use
        self._items: list[MemoryItem] = []

    def retrieve(self, query: str, context: dict[str, Any]) -> list[MemoryItem]:
        if not self._items:
            return []
        scored = []
        for item in self._items:
            s = jaccard(query, item.content)
            if s > 0:
                # Return a copy so we don't mutate the stored item's score.
                scored.append(MemoryItem(
                    id=item.id, content=item.content, type=item.type,
                    metadata=item.metadata, score=s, created_at=item.created_at,
                ))
        scored.sort(key=lambda i: i.score, reverse=True)
        return scored[: self.top_k]

    def update(self, interaction: dict[str, Any]) -> None:
        reflection = self._build_reflection(interaction)
        if not reflection:
            return
        item = MemoryItem.make(
            content=reflection,
            type="reflection",
            scenario=interaction.get("scenario_name"),
            session_id=interaction.get("session_id"),
            task_id=interaction.get("task_id"),
        )
        self._items.append(item)
        if len(self._items) > self.max_reflections:
            self._items = self._items[-self.max_reflections :]

    def reset(self) -> None:
        self._items.clear()

    def export_memory(self) -> dict[str, Any]:
        return {"name": self.name, "items": [i.to_dict() for i in self._items]}

    @staticmethod
    def _build_reflection(interaction: dict[str, Any]) -> str:
        """Deterministic reflection generator.

        Format: "While solving <task>, I violated <rule_ids>. Next time I
        should <improvement_suggestion>."

        Designed to be retrieval-friendly: includes the scenario name, the
        rule pack, and the rule ids so token-overlap retrieval can pick it
        up on related future tasks.
        """
        feedback = interaction.get("feedback") or {}
        task = interaction.get("task") or {}
        eval_criteria = task.get("evaluation_criteria") or {}
        rule_pack = eval_criteria.get("rule_pack", "unknown")

        violated = feedback.get("violated_rules") or []
        suggestions = feedback.get("improvement_suggestions") or []

        if not violated and feedback.get("task_success"):
            return (
                f"On scenario '{interaction.get('scenario_name')}' "
                f"(rule pack {rule_pack}, session "
                f"{interaction.get('session_id')}): all rules satisfied. "
                f"Strategy worked - reuse the same checking discipline."
            )

        suggestion_text = " ".join(suggestions) if suggestions else \
            "review every rule explicitly before concluding."
        violated_text = ", ".join(violated) if violated else "no specific rules"
        return (
            f"On scenario '{interaction.get('scenario_name')}' "
            f"(rule pack {rule_pack}, session "
            f"{interaction.get('session_id')}, task "
            f"{interaction.get('task_id')}): violated rules [{violated_text}]. "
            f"Next time: {suggestion_text}"
        )
