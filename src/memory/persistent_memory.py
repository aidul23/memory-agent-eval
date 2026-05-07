"""PersistentMemory: long-term JSONL-backed structured store.

Persists every entry to disk so memory can survive process restarts (and
so subsequent experiment runs can warm-start from prior knowledge if
desired). The on-disk format is JSONL, one ``MemoryItem`` per line.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ._similarity import jaccard
from .base_memory import BaseMemory, MemoryItem


class PersistentMemory(BaseMemory):
    name = "persistent"

    def __init__(
        self,
        storage_path: str | Path = "data/persistent_memory/persistent_store.jsonl",
        top_k: int = 5,
        similarity: str = "jaccard",
        **_: Any,
    ) -> None:
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.top_k = top_k
        self.similarity = similarity
        self._items: list[MemoryItem] = self._load()

    def _load(self) -> list[MemoryItem]:
        if not self.storage_path.exists():
            return []
        items: list[MemoryItem] = []
        with self.storage_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                    items.append(MemoryItem(**raw))
                except Exception:
                    continue
        return items

    def retrieve(self, query: str, context: dict[str, Any]) -> list[MemoryItem]:
        if not self._items:
            return []
        scored = []
        for item in self._items:
            s = jaccard(query, item.content)
            if s > 0:
                scored.append(MemoryItem(
                    id=item.id, content=item.content, type=item.type,
                    metadata=item.metadata, score=s, created_at=item.created_at,
                ))
        scored.sort(key=lambda i: i.score, reverse=True)
        return scored[: self.top_k]

    def update(self, interaction: dict[str, Any]) -> None:
        task = interaction.get("task") or {}
        feedback = interaction.get("feedback") or {}
        response = interaction.get("response") or {}

        content = (
            f"scenario={task.get('scenario_name')} "
            f"session={task.get('session_id')} "
            f"task={task.get('task_id')} "
            f"description={task.get('input_description', '')} "
            f"decision={response.get('decision', '')} "
            f"violated={feedback.get('violated_rules', [])} "
            f"score={feedback.get('rule_compliance_score', 'n/a')}"
        )
        item = MemoryItem.make(
            content=content,
            type="persistent",
            scenario=task.get("scenario_name"),
            session_id=task.get("session_id"),
            task_id=task.get("task_id"),
            rule_compliance_score=feedback.get("rule_compliance_score"),
        )
        self._items.append(item)
        with self.storage_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(item.to_dict()) + "\n")

    def reset(self) -> None:
        self._items.clear()
        if self.storage_path.exists():
            self.storage_path.unlink()
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

    def export_memory(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "storage_path": str(self.storage_path),
            "items": [i.to_dict() for i in self._items],
        }
