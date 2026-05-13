"""AContext wrapper (self-hosted or hosted).

AContext models memory as sessions of structured messages. We map one
experiment run to one AContext session, so retrieval is naturally scoped.

Local infra (recommended for the experiment suite):

    curl -fsSL https://install.acontext.io | sh
    acontext server up
    # API on http://localhost:8029/api/v1
    # Dashboard on http://localhost:3000/
    # Default API key: sk-ac-your-root-api-bearer-token

Set ``ACONTEXT_API_URL`` and ``ACONTEXT_API_KEY`` in your ``.env`` and the
wrapper auto-connects. If either is missing the wrapper falls back to the
local Jaccard stub.

Retrieval shape
---------------
AContext's primary read path is ``sessions.get_messages`` (returns all
messages of a session). We turn each message into a ``MemoryItem`` and let
``ExternalMemoryBase`` rank them locally with Jaccard so the top_k contract
is honoured. If AContext's skill-search API is enabled on your deployment,
override ``_remote_retrieve`` to use it instead.
"""

from __future__ import annotations

import os
from typing import Any

from ._external_base import ExternalMemoryBase
from .base_memory import MemoryItem
from ._similarity import jaccard


class AContextMemory(ExternalMemoryBase):
    name = "acontext"

    def __init__(
        self,
        workspace: str = "dfx_agent",
        top_k: int = 5,
        **kwargs: Any,
    ) -> None:
        self.workspace = workspace
        self._api_key = kwargs.pop("api_key", None) or os.getenv("ACONTEXT_API_KEY")
        self._api_url = (
            kwargs.pop("base_url", None)
            or os.getenv("ACONTEXT_API_URL")
            or "http://localhost:8029/api/v1"
        )
        # session_id per (run_id, scenario) created lazily inside _remote_update.
        self._session_ids: dict[str, str] = {}
        super().__init__(top_k=top_k, **kwargs)

    def _connect(self) -> Any:
        if not self._api_key:
            raise RuntimeError("ACONTEXT_API_KEY not set")
        try:
            from acontext import AcontextClient  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("acontext SDK not installed (pip install acontext)") from exc
        try:
            client = AcontextClient(api_key=self._api_key, base_url=self._api_url)
            # Best-effort health check; some SDK versions raise if the server is down.
            try:
                client.ping()
            except Exception:
                pass
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"AContext client init failed: {exc}") from exc
        return client

    # ---- Session scoping ---------------------------------------------

    def _session_key(self, ctx: dict[str, Any]) -> str:
        return f"{ctx.get('run_id') or 'no_run'}::{ctx.get('scenario_name') or 'default'}"

    def _get_or_create_session_id(self, ctx: dict[str, Any]) -> str | None:
        if self._client is None:
            return None
        key = self._session_key(ctx)
        if key in self._session_ids:
            return self._session_ids[key]
        try:
            session = self._client.sessions.create(user=self.workspace)
            sid = getattr(session, "id", None) or getattr(session, "session_id", None)
        except Exception:
            return None
        if sid:
            self._session_ids[key] = sid
        return sid

    # ---- Remote API ---------------------------------------------------

    def _remote_retrieve(self, query: str, context: dict[str, Any]) -> list[MemoryItem]:
        if self._client is None:
            return []
        sid = self._session_ids.get(self._session_key(context))
        if not sid:
            return []
        try:
            resp = self._client.sessions.get_messages(session_id=sid)
        except Exception:
            return []
        items = getattr(resp, "items", None) or resp or []
        scored: list[MemoryItem] = []
        for msg in items:
            content = self._extract_text(msg)
            if not content:
                continue
            s = jaccard(query, content)
            if s > 0:
                scored.append(MemoryItem(
                    id=str(getattr(msg, "id", "") or getattr(msg, "message_id", "")),
                    content=content,
                    type="acontext",
                    metadata={"session_id": sid},
                    score=s,
                ))
        scored.sort(key=lambda i: i.score, reverse=True)
        return scored[: self.top_k]

    def _remote_update(self, interaction: dict[str, Any]) -> None:
        if self._client is None:
            return
        sid = self._get_or_create_session_id(interaction)
        if not sid:
            raise RuntimeError("acontext: could not get or create session")
        item = self._build_item(interaction)
        try:
            self._client.sessions.store_message(
                session_id=sid,
                blob={"role": "assistant", "content": item.content},
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"acontext.store_message failed: {exc}") from exc

    def _remote_clear(self) -> None:
        """Best-effort delete of every session we created.

        Transport cleanup is handled by ``ExternalMemoryBase.close()``.
        """
        if self._client is None:
            return
        for sid in list(self._session_ids.values()):
            # AContext exposes session deletion under client.sessions.delete in
            # most SDK versions; fall through silently if missing.
            for method_name in ("delete", "destroy", "remove"):
                fn = getattr(getattr(self._client, "sessions", None), method_name, None)
                if callable(fn):
                    try:
                        fn(session_id=sid)
                    except Exception:  # noqa: BLE001
                        continue
                    break
        self._session_ids.clear()

    @staticmethod
    def _extract_text(msg: Any) -> str:
        """Robustly pull text out of an AContext message in either dict or SDK-object form."""
        if isinstance(msg, dict):
            blob = msg.get("blob") or {}
            return str(blob.get("content") or msg.get("content") or "")
        blob = getattr(msg, "blob", None)
        if isinstance(blob, dict):
            return str(blob.get("content", ""))
        return str(getattr(msg, "content", "") or "")
