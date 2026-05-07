"""Shared scaffolding for external memory provider wrappers.

Every external wrapper (Mem0, Zep, Supermemory, AContext) follows the same
pattern:

1. On construction, attempt to import / connect to the provider's SDK.
2. If the SDK is missing or credentials are absent, fall back to an
   in-memory implementation that is 100% interface-compliant. This keeps
   the platform usable in offline environments.
3. ``retrieve`` and ``update`` proxy to the provider's API when available,
   otherwise to the local fallback.

Subclasses implement two thin hooks:
- ``_connect`` - attempt to set up the SDK client; raise to trigger fallback.
- ``_remote_retrieve`` / ``_remote_update`` - real provider calls.
"""

from __future__ import annotations

import abc
from typing import Any

from ..utils import get_logger
from ._similarity import jaccard
from .base_memory import BaseMemory, MemoryItem

logger = get_logger(__name__)


class ExternalMemoryBase(BaseMemory, abc.ABC):
    """Common base for external-service memory wrappers."""

    name = "external"

    def __init__(self, top_k: int = 5, **kwargs: Any) -> None:
        self.top_k = top_k
        self._kwargs = kwargs
        self._items: list[MemoryItem] = []
        self._client: Any | None = None
        try:
            self._client = self._connect()
            self._mode = "remote"
        except Exception as exc:  # noqa: BLE001
            logger.info(
                "[%s] external client unavailable (%s) - falling back to local stub.",
                self.name, exc,
            )
            self._mode = "local"

    @property
    def mode(self) -> str:
        return self._mode

    # ---- Subclass hooks ----------------------------------------------

    @abc.abstractmethod
    def _connect(self) -> Any:
        """Attempt to set up an SDK client. Raise to trigger local fallback."""

    def _remote_retrieve(self, query: str, context: dict[str, Any]) -> list[MemoryItem]:
        """Provider-specific retrieval. Override for real integration."""
        return []

    def _remote_update(self, interaction: dict[str, Any]) -> None:
        """Provider-specific upsert. Override for real integration."""
        return None

    # ---- BaseMemory implementation -----------------------------------

    def retrieve(self, query: str, context: dict[str, Any]) -> list[MemoryItem]:
        if self._mode == "remote":
            try:
                hits = self._remote_retrieve(query, context)
                if hits:
                    return hits[: self.top_k]
            except Exception as exc:  # noqa: BLE001
                logger.warning("[%s] remote retrieve failed: %s", self.name, exc)
        return self._local_retrieve(query)

    def update(self, interaction: dict[str, Any]) -> None:
        item = self._build_item(interaction)
        self._items.append(item)
        if self._mode == "remote":
            try:
                self._remote_update(interaction)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[%s] remote update failed: %s", self.name, exc)

    def reset(self) -> None:
        self._items.clear()

    def export_memory(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mode": self._mode,
            "items": [i.to_dict() for i in self._items],
        }

    # ---- Helpers -----------------------------------------------------

    def _local_retrieve(self, query: str) -> list[MemoryItem]:
        if not self._items:
            return []
        scored: list[MemoryItem] = []
        for item in self._items:
            s = jaccard(query, item.content)
            if s > 0:
                scored.append(MemoryItem(
                    id=item.id, content=item.content, type=item.type,
                    metadata=item.metadata, score=s, created_at=item.created_at,
                ))
        scored.sort(key=lambda i: i.score, reverse=True)
        return scored[: self.top_k]

    @staticmethod
    def _build_item(interaction: dict[str, Any]) -> MemoryItem:
        task = interaction.get("task") or {}
        feedback = interaction.get("feedback") or {}
        response = interaction.get("response") or {}
        content = (
            f"[{task.get('scenario_name')}::session_{task.get('session_id')}] "
            f"{task.get('input_description', '')} | "
            f"decision={response.get('decision','')} | "
            f"violated={feedback.get('violated_rules', [])}"
        )
        return MemoryItem.make(
            content=content, type="external",
            scenario=task.get("scenario_name"),
            session_id=task.get("session_id"),
            task_id=task.get("task_id"),
        )
