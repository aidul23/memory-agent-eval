"""Vectorize.io Hindsight wrapper (self-hosted recommended).

`Hindsight <https://github.com/vectorize-io/hindsight>`_ is a memory engine
that consolidates facts into "banks" and supports four parallel retrieval
strategies (semantic / BM25 / graph / temporal). It's been benchmarked at
94.6% on LongMemEval.

Local infra
-----------
    docker run -d --name hindsight -p 8888:8888 \\
        -e HINDSIGHT_API_LLM_API_KEY=$OPENAI_API_KEY \\
        ghcr.io/vectorize-io/hindsight

Then set in your ``.env``::

    HINDSIGHT_API_URL=http://localhost:8888
    HINDSIGHT_API_KEY=                       # optional - blank for local

Banks
-----
We create one bank per (run_id, scenario) so each run is isolated. The bank
is created lazily on the first ``update()`` call.

SDK
---
We use the official ``hindsight-client`` package; install with
``pip install hindsight-client``.
"""

from __future__ import annotations

import os
from typing import Any

from ._external_base import ExternalMemoryBase
from .base_memory import MemoryItem


class VectorizeHindsightMemory(ExternalMemoryBase):
    name = "vectorize_hindsight"

    def __init__(
        self,
        bank_prefix: str = "dfx",
        top_k: int = 5,
        budget: str = "mid",       # "low" | "mid" | "high"
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> None:
        self.bank_prefix = bank_prefix
        self.budget = budget
        self.max_tokens = max_tokens
        self._api_url = (
            kwargs.pop("base_url", None)
            or os.getenv("HINDSIGHT_API_URL")
            or "http://localhost:8888"
        )
        self._api_key = kwargs.pop("api_key", None) or os.getenv("HINDSIGHT_API_KEY")
        self._known_banks: set[str] = set()
        super().__init__(top_k=top_k, **kwargs)

    def _connect(self) -> Any:
        try:
            from hindsight_client import Hindsight  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "hindsight-client not installed (pip install hindsight-client)"
            ) from exc
        try:
            kwargs: dict[str, Any] = {"base_url": self._api_url, "timeout": 30.0}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            client = Hindsight(**kwargs)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Hindsight client init failed: {exc}") from exc
        return client

    # ---- Bank scoping -------------------------------------------------

    def _bank_id(self, ctx: dict[str, Any]) -> str:
        scenario = ctx.get("scenario_name") or "default"
        run_id = ctx.get("run_id") or "no_run"
        # bank_id allowed chars are conservative; use ascii + dashes.
        raw = f"{self.bank_prefix}--{run_id}--{scenario}"
        return raw.lower().replace("_", "-").replace(":", "-")

    def _ensure_bank(self, bank_id: str) -> None:
        if self._client is None or bank_id in self._known_banks:
            return
        try:
            self._client.create_bank(
                bank_id=bank_id,
                name=f"DFx eval bank {bank_id}",
                background=(
                    "Bank for the memory-agent-eval research platform. "
                    "Stores feedback-grounded reflections from DFx tasks."
                ),
            )
        except Exception:
            # 409 / already-exists is fine.
            pass
        self._known_banks.add(bank_id)

    # ---- Remote API ---------------------------------------------------

    def _remote_retrieve(self, query: str, context: dict[str, Any]) -> list[MemoryItem]:
        if self._client is None:
            return []
        bank_id = self._bank_id(context)
        try:
            resp = self._client.recall(
                bank_id=bank_id,
                query=query,
                budget=self.budget,
                max_tokens=self.max_tokens,
            )
        except Exception:
            return []
        # `recall()` may return a list directly or a response object with `.results`.
        results = getattr(resp, "results", None)
        if results is None:
            results = list(resp or [])
        out: list[MemoryItem] = []
        for r in results[: self.top_k]:
            text = getattr(r, "text", None) or (r.get("text") if isinstance(r, dict) else None) or ""
            rtype = getattr(r, "type", None) or (r.get("type") if isinstance(r, dict) else None) or "hindsight"
            score = float(getattr(r, "score", 0.0) or (r.get("score", 0.0) if isinstance(r, dict) else 0.0))
            out.append(MemoryItem(
                id=str(getattr(r, "id", "") or (r.get("id") if isinstance(r, dict) else "") or text[:24]),
                content=str(text),
                type=str(rtype),
                metadata={"bank_id": bank_id},
                score=score,
            ))
        return out

    def _remote_update(self, interaction: dict[str, Any]) -> None:
        if self._client is None:
            return
        bank_id = self._bank_id(interaction)
        self._ensure_bank(bank_id)
        item = self._build_item(interaction)
        task = interaction.get("task") or {}
        # Hindsight's MemoryItem.metadata is dict[str, str] - coerce every value
        # (session_id is often an int in our task schema) to avoid pydantic
        # validation errors on the server side.
        metadata = {
            k: str(v)
            for k, v in {
                "scenario": task.get("scenario_name"),
                "session_id": task.get("session_id"),
                "task_id": task.get("task_id"),
                "run_id": interaction.get("run_id"),
            }.items()
            if v is not None
        }
        try:
            self._client.retain(
                bank_id=bank_id,
                content=item.content,
                context=f"scenario={task.get('scenario_name')} session={task.get('session_id')}",
                document_id=str(task.get("task_id", "") or ""),
                metadata=metadata,
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"hindsight.retain failed: {exc}") from exc

    def _remote_clear(self) -> None:
        """Delete every bank we created in this process.

        Transport cleanup (closing the aiohttp session) is handled by
        ``ExternalMemoryBase.close()``; callers should use the context
        manager or invoke ``close()`` directly.
        """
        if self._client is None:
            return
        for bank_id in list(self._known_banks):
            fn = getattr(self._client, "delete_bank", None)
            if callable(fn):
                try:
                    fn(bank_id=bank_id)
                except Exception:  # noqa: BLE001
                    pass
        self._known_banks.clear()
