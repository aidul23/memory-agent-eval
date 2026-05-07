"""StatelessMemory: the no-memory baseline."""

from __future__ import annotations

from typing import Any

from .base_memory import BaseMemory, MemoryItem


class StatelessMemory(BaseMemory):
    """Stores nothing, retrieves nothing.

    Used as the experimental baseline. ``update`` is a no-op so that the
    agent control flow stays uniform across all memory variants.
    """

    name = "stateless"

    def __init__(self, **_: Any) -> None:
        pass

    def retrieve(self, query: str, context: dict[str, Any]) -> list[MemoryItem]:
        return []

    def update(self, interaction: dict[str, Any]) -> None:
        return None

    def reset(self) -> None:
        return None

    def export_memory(self) -> dict[str, Any]:
        return {"name": self.name, "items": []}
