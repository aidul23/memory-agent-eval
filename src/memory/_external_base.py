"""Shared scaffolding for external memory provider wrappers.

Every external wrapper (Mem0, Zep, AContext, Vectorize Hindsight) follows
the same pattern:

1. On construction, attempt to import / connect to the provider's SDK.
2. If the SDK is missing or credentials are absent, fall back to an
   in-memory implementation that is 100% interface-compliant. This keeps
   the platform usable in offline environments.
3. ``retrieve``, ``update``, and ``reset`` proxy to the provider's API when
   available, otherwise to the local fallback.

Reset policy
------------
Different research questions need different reset semantics. ``reset_policy``
on each wrapper controls what happens when the runner calls ``reset()``:

- ``clear_remote`` (default) - wipe both the local mirror AND the data
  stored in the remote service. Use this when each experimental run must
  start from a clean slate (standard isolation for statistical tests).
- ``keep_remote`` - clear only the local mirror; remote data persists.
  Use this when studying long-term learning across runs.
- ``clear_local_only`` - alias of ``keep_remote`` but explicit; useful for
  debugging or when the remote API has no delete endpoint.

Subclasses implement four thin hooks:
- ``_connect`` - attempt to set up the SDK client; raise to trigger fallback.
- ``_remote_retrieve`` / ``_remote_update`` - real provider calls.
- ``_remote_clear`` - delete the relevant remote partition (per-scenario,
  per-collection, etc.). Optional; default is a no-op + warning.
"""

from __future__ import annotations

import abc
from typing import Any, Literal

from ..utils import get_logger
from ._similarity import jaccard
from .base_memory import BaseMemory, MemoryItem

logger = get_logger(__name__)

ResetPolicy = Literal["clear_remote", "keep_remote", "clear_local_only"]
_VALID_POLICIES: tuple[ResetPolicy, ...] = ("clear_remote", "keep_remote", "clear_local_only")


class ExternalMemoryBase(BaseMemory, abc.ABC):
    """Common base for external-service memory wrappers."""

    name = "external"

    def __init__(
        self,
        top_k: int = 5,
        reset_policy: ResetPolicy = "clear_remote",
        **kwargs: Any,
    ) -> None:
        if reset_policy not in _VALID_POLICIES:
            raise ValueError(
                f"reset_policy must be one of {_VALID_POLICIES}, got {reset_policy!r}"
            )
        self.top_k = top_k
        self.reset_policy: ResetPolicy = reset_policy
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

    def _remote_clear(self) -> None:
        """Provider-specific delete. Override for real integration.

        Default behaviour logs a warning and does nothing - safe for
        wrappers that have no native delete or are still being implemented.
        """
        logger.warning(
            "[%s] _remote_clear() not implemented - remote data not deleted on reset.",
            self.name,
        )

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
        # Local mirror is always cleared - it's just a Python list.
        self._items.clear()
        if self._mode != "remote":
            return
        if self.reset_policy in {"keep_remote", "clear_local_only"}:
            logger.info("[%s] reset_policy=%s - remote data kept.",
                        self.name, self.reset_policy)
            return
        # reset_policy == "clear_remote"
        try:
            self._remote_clear()
            logger.info("[%s] remote data cleared (reset_policy=clear_remote).", self.name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] remote clear failed: %s", self.name, exc)

    def close(self) -> None:
        """Release transport resources held by the SDK client.

        Most SDK clients use httpx/aiohttp under the hood and emit
        "Unclosed client session" warnings on garbage collection if we
        never explicitly close them. Subclasses that hold extra
        resources can override.
        """
        client = self._client
        if client is None:
            return
        for method in ("close", "aclose"):
            fn = getattr(client, method, None)
            if not callable(fn):
                continue
            try:
                result = fn()
            except Exception:  # noqa: BLE001
                continue
            # If the SDK exposes aclose() returning a coroutine, run it.
            if result is not None and hasattr(result, "__await__"):
                try:
                    import asyncio
                    asyncio.get_event_loop().run_until_complete(result)
                except Exception:  # noqa: BLE001
                    pass
            break
        self._client = None

    def __enter__(self) -> "ExternalMemoryBase":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def export_memory(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mode": self._mode,
            "reset_policy": self.reset_policy,
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
            run_id=interaction.get("run_id"),
        )
