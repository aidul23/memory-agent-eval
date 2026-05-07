"""Supermemory wrapper.

Real integration sketch using Supermemory's REST API:

    POST /v1/memories      body: {content, collection}
    GET  /v1/memories/search?q=&collection=

This wrapper falls back to a local in-memory store when no API key is
present, so the rest of the pipeline can still be exercised offline.
"""

from __future__ import annotations

import os
from typing import Any

from ._external_base import ExternalMemoryBase


class SupermemoryMemory(ExternalMemoryBase):
    name = "supermemory"

    def __init__(
        self,
        collection: str = "dfx_agent",
        top_k: int = 5,
        **kwargs: Any,
    ) -> None:
        self.collection = collection
        self._api_key = kwargs.pop("api_key", None) or os.getenv("SUPERMEMORY_API_KEY")
        self._base_url = kwargs.pop("base_url", "https://api.supermemory.ai")
        super().__init__(top_k=top_k, **kwargs)

    def _connect(self) -> Any:
        if not self._api_key:
            raise RuntimeError("SUPERMEMORY_API_KEY not set")
        # Real impl would create a session here.
        return {"base_url": self._base_url, "api_key": self._api_key}
