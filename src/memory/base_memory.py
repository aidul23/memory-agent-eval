"""Memory interface used by every memory implementation.

The contract (retrieve / update / reset / export_memory) is intentionally
narrow so wrappers around external services (Mem0, Zep, Supermemory,
AContext) only need to translate between this surface and their own SDK.
"""

from __future__ import annotations

import abc
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class MemoryItem:
    """A single entry stored in (or returned from) a memory."""

    id: str
    content: str
    type: str = "generic"            # 'reflection' | 'context' | 'fact' | ...
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0               # populated on retrieval
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def make(content: str, type: str = "generic", **metadata: Any) -> "MemoryItem":
        return MemoryItem(
            id=str(uuid.uuid4()),
            content=content,
            type=type,
            metadata=metadata,
        )


class BaseMemory(abc.ABC):
    """Abstract memory interface."""

    name: str = "base"

    @abc.abstractmethod
    def retrieve(self, query: str, context: dict[str, Any]) -> list[MemoryItem]:
        """Return the most relevant memory items for the given query."""

    @abc.abstractmethod
    def update(self, interaction: dict[str, Any]) -> None:
        """Ingest a completed agent interaction (task + response + feedback)."""

    @abc.abstractmethod
    def reset(self) -> None:
        """Drop all stored items - usually called between experiment runs."""

    @abc.abstractmethod
    def export_memory(self) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot of every stored item."""

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.__class__.__name__}()"
