"""Zep wrapper.

Real integration sketch:

    from zep_python.client import Zep
    self._client = Zep(api_key=..., base_url=...)
    self._client.memory.add(session_id=..., messages=[...])
    self._client.memory.search_sessions(text=..., session_ids=[...])
"""

from __future__ import annotations

import os
from typing import Any

from ._external_base import ExternalMemoryBase
from .base_memory import MemoryItem


class ZepMemory(ExternalMemoryBase):
    name = "zep"

    def __init__(
        self,
        session_id_prefix: str = "dfx",
        top_k: int = 5,
        **kwargs: Any,
    ) -> None:
        self.session_id_prefix = session_id_prefix
        self._api_url = kwargs.pop("api_url", None) or os.getenv("ZEP_API_URL")
        self._api_key = kwargs.pop("api_key", None) or os.getenv("ZEP_API_KEY")
        super().__init__(top_k=top_k, **kwargs)

    def _connect(self) -> Any:
        if not self._api_url:
            raise RuntimeError("ZEP_API_URL not set")
        try:
            from zep_python.client import Zep  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("zep-python SDK not installed") from exc
        return Zep(api_key=self._api_key, base_url=self._api_url)

    def _remote_retrieve(self, query: str, context: dict[str, Any]) -> list[MemoryItem]:
        if self._client is None:
            return []
        try:
            sid = f"{self.session_id_prefix}_{context.get('scenario_name','default')}"
            hits = self._client.memory.search_sessions(text=query, session_ids=[sid])
        except Exception:
            return []
        out: list[MemoryItem] = []
        for h in hits or []:
            content = getattr(h, "message", None) or getattr(h, "text", "") or ""
            out.append(MemoryItem(
                id=str(getattr(h, "uuid", "")),
                content=str(content),
                type="zep",
                metadata={"raw": str(h)},
                score=float(getattr(h, "dist", 0.0) or 0.0),
            ))
        return out

    def _remote_update(self, interaction: dict[str, Any]) -> None:
        if self._client is None:
            return
        item = self._build_item(interaction)
        sid = f"{self.session_id_prefix}_{interaction.get('scenario_name','default')}"
        self._client.memory.add(
            session_id=sid,
            messages=[{"role": "system", "content": item.content}],
        )
