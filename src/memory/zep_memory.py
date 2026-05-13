"""Zep wrapper (self-hosted Community Edition or hosted Zep Cloud).

Uses the ``zep-cloud`` Python SDK (>=3.x) against either:

- a local Zep Community Edition deployment (``ZEP_API_URL=http://localhost:8000``), or
- Zep Cloud (``ZEP_API_URL=https://api.getzep.com`` + ``ZEP_API_KEY=...``).

API surface
-----------
``zep-cloud`` 3.x reorganised the API:

- ``client.user.add(...)``      - create/upsert a user (one per memory instance)
- ``client.thread.create(...)`` - create a thread (one per run+scenario)
- ``client.thread.add_messages(thread_id, messages=[Message(...)])``
- ``client.thread.get_user_context(thread_id)`` - condensed retrieval
- ``client.thread.delete(thread_id)`` - cleanup

The old ``client.memory.*`` surface no longer exists, hence the rewrite.

Threads
-------
We map one (run_id, scenario) pair to one Zep thread so different runs of the
same condition cannot contaminate each other:

    thread_id = f"{prefix}__{run_id}__{scenario}"
    user_id   = prefix   (one shared user per ZepMemory instance)
"""

from __future__ import annotations

import os
from typing import Any

from ..utils import get_logger
from ._external_base import ExternalMemoryBase
from .base_memory import MemoryItem

logger = get_logger(__name__)


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
        # Community Edition often runs with no API key. Use a sentinel so the
        # SDK does not crash on auth header construction.
        self._api_key = (
            kwargs.pop("api_key", None)
            or os.getenv("ZEP_API_KEY")
            or "zep-community-edition"
        )
        # Threads we've written to in this process (for cleanup on reset).
        self._known_threads: set[str] = set()
        # Whether we've already attempted user-creation this session.
        self._user_ready: bool = False
        super().__init__(top_k=top_k, **kwargs)

    def _connect(self) -> Any:
        if not self._api_url:
            raise RuntimeError("ZEP_API_URL not set")
        try:
            from zep_cloud.client import Zep  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("zep-cloud SDK not installed (pip install zep-cloud)") from exc
        try:
            client = Zep(api_key=self._api_key, base_url=self._api_url)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Zep client init failed: {exc}") from exc
        return client

    # ---- Scoping ------------------------------------------------------

    def _thread_id(self, ctx: dict[str, Any]) -> str:
        scenario = ctx.get("scenario_name") or "default"
        run_id = ctx.get("run_id") or "no_run"
        return f"{self.session_id_prefix}__{run_id}__{scenario}"

    def _user_id(self) -> str:
        return self.session_id_prefix

    def _ensure_user(self) -> None:
        """Create the shared user idempotently (Zep raises 409 if exists)."""
        if self._client is None or self._user_ready:
            return
        try:
            self._client.user.add(user_id=self._user_id())
        except Exception as exc:  # noqa: BLE001
            # 409 conflict (already exists) is fine; log everything else once.
            logger.debug("[zep] user.add(%s) -> %s", self._user_id(), exc)
        self._user_ready = True

    def _ensure_thread(self, thread_id: str) -> None:
        """Create the thread idempotently."""
        if self._client is None or thread_id in self._known_threads:
            return
        self._ensure_user()
        try:
            self._client.thread.create(thread_id=thread_id, user_id=self._user_id())
        except Exception as exc:  # noqa: BLE001
            # 409 conflict is fine; log everything else for debugging.
            logger.debug("[zep] thread.create(%s) -> %s", thread_id, exc)
        self._known_threads.add(thread_id)

    # ---- Remote API ---------------------------------------------------

    def _remote_retrieve(self, query: str, context: dict[str, Any]) -> list[MemoryItem]:
        if self._client is None:
            return []
        tid = self._thread_id(context)
        try:
            resp = self._client.thread.get_user_context(thread_id=tid)
        except Exception:
            return []
        # ThreadContextResponse exposes a `context` string (the digested view).
        ctx_str = getattr(resp, "context", None) or getattr(resp, "summary", None)
        if not ctx_str:
            return []
        return [
            MemoryItem(
                id=tid,
                content=str(ctx_str),
                type="zep",
                metadata={"thread_id": tid},
                score=1.0,
            )
        ]

    def _remote_update(self, interaction: dict[str, Any]) -> None:
        if self._client is None:
            return
        item = self._build_item(interaction)
        tid = self._thread_id(interaction)
        self._ensure_thread(tid)

        try:
            from zep_cloud.types.message import Message  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("zep-cloud SDK not installed") from exc

        messages = [Message(role="assistant", content=item.content, name="dfx-agent")]
        try:
            self._client.thread.add_messages(thread_id=tid, messages=messages)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"zep thread.add_messages failed: {exc}") from exc

    def _remote_clear(self) -> None:
        """Delete every thread we created in this process.

        Transport cleanup is handled by ``ExternalMemoryBase.close()``.
        """
        if self._client is None:
            return
        for tid in list(self._known_threads):
            try:
                self._client.thread.delete(thread_id=tid)
            except Exception:  # noqa: BLE001
                continue
        self._known_threads.clear()
