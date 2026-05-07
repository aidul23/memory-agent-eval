"""ContextualMemory: stores task context, decisions, and feedback.

Unlike HindsightMemory which only stores *reflections*, ContextualMemory
keeps richer entries (full task description, agent decision, structured
feedback). At retrieval time it scores entries by token similarity against
the current task description plus its design context.
"""

from __future__ import annotations

import json
from typing import Any

from ._similarity import jaccard
from .base_memory import BaseMemory, MemoryItem


class ContextualMemory(BaseMemory):
    name = "contextual"

    def __init__(
        self,
        max_entries: int = 200,
        top_k: int = 5,
        similarity: str = "jaccard",
        **_: Any,
    ) -> None:
        self.max_entries = max_entries
        self.top_k = top_k
        self.similarity = similarity
        self._items: list[MemoryItem] = []

    def retrieve(self, query: str, context: dict[str, Any]) -> list[MemoryItem]:
        if not self._items:
            return []
        # Augment the query with serialised context so design parameters
        # (material, dimensions) participate in retrieval.
        ctx_text = json.dumps(context, sort_keys=True)
        full_query = f"{query}\n{ctx_text}"
        scored: list[MemoryItem] = []
        for item in self._items:
            s = jaccard(full_query, item.content)
            if s > 0:
                scored.append(MemoryItem(
                    id=item.id, content=item.content, type=item.type,
                    metadata=item.metadata, score=s, created_at=item.created_at,
                ))
        scored.sort(key=lambda i: i.score, reverse=True)
        return scored[: self.top_k]

    def update(self, interaction: dict[str, Any]) -> None:
        task = interaction.get("task") or {}
        decision = (interaction.get("response") or {}).get("decision", "")
        feedback = interaction.get("feedback") or {}

        snippet = (
            f"[{task.get('scenario_name')}::session_{task.get('session_id')}] "
            f"{task.get('input_description', '')} | "
            f"context={json.dumps(task.get('design_context', {}), sort_keys=True)} | "
            f"decision={decision} | "
            f"violated={feedback.get('violated_rules', [])} | "
            f"score={feedback.get('rule_compliance_score', 'n/a')}"
        )

        item = MemoryItem.make(
            content=snippet,
            type="context",
            scenario=task.get("scenario_name"),
            session_id=task.get("session_id"),
            task_id=task.get("task_id"),
            rule_compliance_score=feedback.get("rule_compliance_score"),
        )
        self._items.append(item)
        if len(self._items) > self.max_entries:
            self._items = self._items[-self.max_entries :]

    def reset(self) -> None:
        self._items.clear()

    def export_memory(self) -> dict[str, Any]:
        return {"name": self.name, "items": [i.to_dict() for i in self._items]}
