"""Mem0 wrapper - supports both Mem0 Platform (hosted) and Mem0 OSS.

Mode selection
--------------
- If ``MEM0_API_KEY`` is set (or passed in ``api_key=``), the wrapper uses
  the hosted ``MemoryClient`` and talks to mem0.ai over HTTPS. No local
  infra required.
- Otherwise it falls back to the OSS ``Memory`` class, which runs the
  vector store in-process against Qdrant on ``$MEM0_QDRANT_URL`` (default
  ``http://localhost:6333``). Bring Qdrant up first - see
  ``docker compose --profile mem0-oss up -d qdrant``.

Common API surface
------------------
Both ``Memory`` (OSS) and ``MemoryClient`` (hosted) expose ``add``,
``search``, ``delete_all`` with compatible kwargs once you route the
caller through ``filters={"user_id": ...}`` (the hosted client rejects
``user_id`` as a top-level search arg in v2.x), so the implementation
below is mode-agnostic.

The wrapper falls back to the in-RAM Jaccard stub from
``ExternalMemoryBase`` when:
- ``mem0ai`` is not installed, or
- the chosen mode cannot initialise its client (e.g. Qdrant unreachable,
  invalid hosted API key).
"""

from __future__ import annotations

import os
from typing import Any

from ..utils import get_logger
from ._external_base import ExternalMemoryBase
from .base_memory import MemoryItem

logger = get_logger(__name__)


def _build_oss_config() -> dict[str, Any]:
    """Construct a Mem0-OSS config dict from environment variables."""
    qdrant_url = os.getenv("MEM0_QDRANT_URL", "http://localhost:6333")
    llm_provider = os.getenv("MEM0_LLM_PROVIDER", "openai")
    llm_model = os.getenv("MEM0_LLM_MODEL", "gpt-4o-mini")
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
        api_key: str | None = None,
        host: str | None = None,
        infer: bool = False,
        **kwargs: Any,
    ) -> None:
        self.user_id = user_id
        self._config_override = config
        self._api_key = api_key or os.getenv("MEM0_API_KEY") or None
        self._host = host or os.getenv("MEM0_API_HOST") or None
        # `infer=True` asks Mem0 to run an async LLM extraction over each
        # added message and store only the distilled facts. For tight eval
        # loops where retrieve() is called seconds after update(), the
        # extraction often hasn't completed (or has returned no facts for
        # our terse experience records) and the search yields nothing.
        # Default to `infer=False` so messages are stored verbatim and are
        # immediately searchable, which matches how the other backends
        # (Hindsight, Zep, etc.) behave. Override per-instance for
        # research questions that explicitly want LLM-distilled storage.
        self._infer = bool(infer)
        # The agent's retrieve-side context does not include run_id, but the
        # update-side interaction does. Cache the latter so subsequent reads
        # target the same scoped user_id and hit the same partition.
        self._current_run_id: str | None = None
        # "hosted" or "oss" - set in _connect; useful for logging.
        self._kind: str | None = None
        super().__init__(top_k=top_k, **kwargs)

    @property
    def kind(self) -> str | None:
        """Which Mem0 backend is in use ("hosted" or "oss")."""
        return self._kind

    # ---- Connection --------------------------------------------------

    def _connect(self) -> Any:
        try:
            from mem0 import Memory, MemoryClient  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("mem0ai SDK not installed (pip install mem0ai)") from exc

        # Prefer hosted when an API key is available - no local infra needed.
        if self._api_key:
            try:
                kwargs: dict[str, Any] = {"api_key": self._api_key}
                if self._host:
                    kwargs["host"] = self._host
                client = MemoryClient(**kwargs)
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"Mem0 hosted init failed: {exc}") from exc
            self._kind = "hosted"
            logger.info("[%s] using hosted Mem0 Platform.", self.name)
            return client

        # OSS path: needs Qdrant up and OPENAI_API_KEY set for extraction.
        cfg = self._config_override or _build_oss_config()
        try:
            client = Memory.from_config(cfg)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Mem0 OSS init failed: {exc}") from exc
        self._kind = "oss"
        logger.info(
            "[%s] using Mem0 OSS (Qdrant at %s).",
            self.name,
            cfg.get("vector_store", {}).get("config", {}).get("url"),
        )
        return client

    # ---- Scoping -----------------------------------------------------

    def _scoped_user_id(self, ctx: dict[str, Any] | None = None) -> str:
        """Per-run isolation: prepend run_id so different runs do not mix.

        Reads run_id from the caller's dict OR from the cached value
        populated by ``_remote_update``. Falls back to the plain
        ``user_id`` when no run context is available (e.g. interactive
        ``run-single`` calls).
        """
        ctx = ctx or {}
        task = ctx.get("task") or {}
        run_id = (
            ctx.get("run_id")
            or task.get("run_id")
            or self._current_run_id
        )
        return f"{self.user_id}__{run_id}" if run_id else self.user_id

    # ---- Remote API --------------------------------------------------

    def _remote_retrieve(self, query: str, context: dict[str, Any]) -> list[MemoryItem]:
        if self._client is None:
            return []
        user_id = self._scoped_user_id(context)
        # Route user_id via filters - works on both hosted and OSS, and the
        # hosted client explicitly rejects user_id as a top-level kwarg
        # in v2.x.
        try:
            resp = self._client.search(
                query=query,
                filters={"user_id": user_id},
                top_k=self.top_k,
            )
        except TypeError:
            # Very old OSS versions used `limit` instead of `top_k`.
            resp = self._client.search(
                query=query,
                filters={"user_id": user_id},
                limit=self.top_k,
            )
        results = resp.get("results", []) if isinstance(resp, dict) else list(resp or [])
        out: list[MemoryItem] = []
        for r in results:
            if not isinstance(r, dict):
                continue
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
        rid = interaction.get("run_id") or (interaction.get("task") or {}).get("run_id")
        if rid:
            self._current_run_id = str(rid)
        item = self._build_item(interaction)
        task = interaction.get("task") or {}
        metadata = {
            "scenario": task.get("scenario_name"),
            "session_id": task.get("session_id"),
            "task_id": task.get("task_id"),
            "run_id": interaction.get("run_id"),
        }
        messages = [{"role": "user", "content": item.content}]
        try:
            self._client.add(
                messages,
                user_id=self._scoped_user_id(interaction),
                metadata=metadata,
                infer=self._infer,
            )
        except Exception as exc:  # noqa: BLE001
            # Re-raise so ExternalMemoryBase logs it as a warning and
            # falls through to local-only.
            raise RuntimeError(f"mem0.add failed: {exc}") from exc

    def _remote_clear(self) -> None:
        """Delete every memory belonging to this experiment's user_ids.

        ``_known_users`` tracks every scoped variant we observed during
        this wrapper's lifetime AND the cached current_run_id, so reset()
        between runs is symmetric across hosted/OSS.
        """
        if self._client is None:
            return
        users_to_clear: set[str] = {self.user_id}
        if self._current_run_id:
            users_to_clear.add(f"{self.user_id}__{self._current_run_id}")
        for item in self._items:
            ru = (item.metadata or {}).get("run_id")
            if ru:
                users_to_clear.add(f"{self.user_id}__{ru}")
        for uid in users_to_clear:
            try:
                self._client.delete_all(user_id=uid)
            except Exception as exc:  # noqa: BLE001
                logger.debug("[%s] delete_all(user_id=%s) failed: %s", self.name, uid, exc)
                continue
