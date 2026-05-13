"""Mem0 wrapper (self-hostable).

Mem0 is a Python library, not a server. It runs in-process and stores
vectors in an external store (Qdrant by default). For this platform we
expect the user to have Qdrant running locally on
``$MEM0_QDRANT_URL`` (default ``http://localhost:6333``) and an
``OPENAI_API_KEY`` set for the memory-extraction LLM. See README.md
"Infrastructure setup" for the Docker incantations.

Falls back to a local Jaccard stub when:
- ``mem0ai`` is not installed, or
- ``Memory.from_config`` cannot reach the vector store.
"""

from __future__ import annotations

import os
from typing import Any

from ._external_base import ExternalMemoryBase
from .base_memory import MemoryItem


def _build_default_config() -> dict[str, Any]:
    """Construct a Mem0 config dict from environment variables."""
    qdrant_url = os.getenv("MEM0_QDRANT_URL", "http://localhost:6333")
    llm_provider = os.getenv("MEM0_LLM_PROVIDER", "openai")
    llm_model = os.getenv("MEM0_LLM_MODEL", "gpt-4o-mini")
    # Mem0 accepts a "url" or split "host"/"port". URL is simplest.
    return {
        "vector_store": {
            "provider": os.getenv("MEM0_VECTOR_STORE", "qdrant"),
            "config": {
                "url": qdrant_url,
                "collection_name": os.getenv("MEM0_COLLECTION", "dfx_agent_memory"),
            },
        },
        "llm": {
            "provider": llm_provider,
            "config": {"model": llm_model},
        },
    }


class Mem0Memory(ExternalMemoryBase):
    name = "mem0"

    def __init__(
        self,
        user_id: str = "dfx_agent",
        top_k: int = 5,
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.user_id = user_id
        self._config_override = config
        super().__init__(top_k=top_k, **kwargs)

    def _connect(self) -> Any:
        try:
            from mem0 import Memory  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("mem0ai SDK not installed (pip install mem0ai)") from exc
        cfg = self._config_override or _build_default_config()
        try:
            client = Memory.from_config(cfg)
        except Exception as exc:  # noqa: BLE001
            # Most commonly: Qdrant not reachable, or OPENAI_API_KEY missing.
            raise RuntimeError(f"Mem0 init failed: {exc}") from exc
        return client

    # ---- Remote API ---------------------------------------------------

    def _scoped_user_id(self, interaction: dict[str, Any] | None = None) -> str:
        """Per-run isolation: prepend run_id so different runs do not mix.

        Falls back to the plain ``user_id`` if no run context is available
        (e.g. interactive ``run-single`` calls).
        """
        run_id = (interaction or {}).get("run_id")
        return f"{self.user_id}__{run_id}" if run_id else self.user_id

    def _remote_retrieve(self, query: str, context: dict[str, Any]) -> list[MemoryItem]:
        if self._client is None:
            return []
        user_id = self._scoped_user_id(context)
        try:
            resp = self._client.search(
                query=query,
                user_id=user_id,
                limit=self.top_k,
            )
        except TypeError:
            # Older Mem0 versions used `filters={"user_id": ...}` instead of user_id=
            resp = self._client.search(query=query, filters={"user_id": user_id}, limit=self.top_k)
        results = resp.get("results", []) if isinstance(resp, dict) else list(resp or [])
        out: list[MemoryItem] = []
        for r in results:
            out.append(MemoryItem(
                id=str(r.get("id", "")),
                content=str(r.get("memory", r.get("text", ""))),
                type="mem0",
                metadata={k: v for k, v in r.items()
                          if k not in {"id", "memory", "text", "score"}},
                score=float(r.get("score", 0.0) or 0.0),
            ))
        return out

    def _remote_update(self, interaction: dict[str, Any]) -> None:
        if self._client is None:
            return
        item = self._build_item(interaction)
        task = interaction.get("task") or {}
        metadata = {
            "scenario": task.get("scenario_name"),
            "session_id": task.get("session_id"),
            "task_id": task.get("task_id"),
            "run_id": interaction.get("run_id"),
        }
        messages = [{"role": "system", "content": item.content}]
        try:
            self._client.add(
                messages,
                user_id=self._scoped_user_id(interaction),
                metadata=metadata,
            )
        except Exception as exc:  # noqa: BLE001
            # Re-raise so ExternalMemoryBase logs it as a warning and
            # falls through to local-only.
            raise RuntimeError(f"mem0.add failed: {exc}") from exc

    def _remote_clear(self) -> None:
        """Delete every memory belonging to the scoped user_id.

        Mem0 exposes ``delete_all(user_id=...)``; we call it for the
        unscoped base user_id AND every run-scoped variant we have
        observed in this process.
        """
        if self._client is None:
            return
        users_to_clear = {self.user_id}
        for item in self._items:
            ru = (item.metadata or {}).get("run_id")
            if ru:
                users_to_clear.add(f"{self.user_id}__{ru}")
        for uid in users_to_clear:
            try:
                self._client.delete_all(user_id=uid)
            except Exception:  # noqa: BLE001
                # Older SDKs may not have delete_all; best-effort.
                continue
        # Transport cleanup is handled by ExternalMemoryBase.close().
