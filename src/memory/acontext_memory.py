"""AContext wrapper (placeholder).

The platform expects AContext to expose a workspace-scoped memory similar
to Mem0. Until a real SDK is wired up the wrapper falls back to the local
stub from ``_external_base.py``.
"""

from __future__ import annotations

import os
from typing import Any

from ._external_base import ExternalMemoryBase


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
        super().__init__(top_k=top_k, **kwargs)

    def _connect(self) -> Any:
        if not self._api_key:
            raise RuntimeError("ACONTEXT_API_KEY not set")
        return {"workspace": self.workspace, "api_key": self._api_key}
