"""Mem0 wrapper.

Real integration:
    from mem0 import Memory
    self._client = Memory()
    self._client.add(messages, user_id=self.user_id)
    self._client.search(query, user_id=self.user_id)

Until those calls are wired up the wrapper falls back to the local stub
implemented in ``_external_base.py``.
"""

from __future__ import annotations

import os
from typing import Any

from ._external_base import ExternalMemoryBase
from .base_memory import MemoryItem


class Mem0Memory(ExternalMemoryBase):
    name = "mem0"

    def __init__(self, user_id: str = "dfx_agent", top_k: int = 5, **kwargs: Any) -> None:
        self.user_id = user_id
        self._api_key = kwargs.pop("api_key", None) or os.getenv("MEM0_API_KEY")
        super().__init__(top_k=top_k, **kwargs)

    def _connect(self) -> Any:
        # Force fallback unless an API key is present. Real users would also
        # `pip install mem0ai` - we leave that out of requirements.txt.
        if not self._api_key:
            raise RuntimeError("MEM0_API_KEY not set")
        try:
            from mem0 import Memory  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("mem0ai SDK not installed") from exc
        return Memory()

    def _remote_retrieve(self, query: str, context: dict[str, Any]) -> list[MemoryItem]:
        if self._client is None:
            return []
        try:
            hits = self._client.search(query, user_id=self.user_id)
        except Exception:
            return []
        out: list[MemoryItem] = []
        for h in hits or []:
            out.append(MemoryItem(
                id=str(h.get("id", "")),
                content=str(h.get("memory", h.get("text", ""))),
                type="mem0",
                metadata={k: v for k, v in h.items() if k not in {"id", "memory", "text"}},
                score=float(h.get("score", 0.0) or 0.0),
            ))
        return out

    def _remote_update(self, interaction: dict[str, Any]) -> None:
        if self._client is None:
            return
        item = self._build_item(interaction)
        self._client.add([{"role": "system", "content": item.content}], user_id=self.user_id)
