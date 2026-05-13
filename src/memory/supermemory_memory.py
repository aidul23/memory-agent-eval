"""Supermemory wrapper - DEPRECATED.

Supermemory does not currently offer a fully self-hostable deployment
(their stack depends on Cloudflare Workers/R2/KV). To keep the experiment
suite "self-hosted only", Supermemory has been dropped from the active
registry. This file is preserved so importers do not crash; the class
raises ``NotImplementedError`` if instantiated.

To re-enable Supermemory, switch to its hosted SaaS (set
``SUPERMEMORY_API_KEY``) and re-register the class in
``src/memory/__init__.py::_REGISTRY``.
"""

from __future__ import annotations

from typing import Any

from ._external_base import ExternalMemoryBase


class SupermemoryMemory(ExternalMemoryBase):
    name = "supermemory"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "SupermemoryMemory is currently disabled because Supermemory has "
            "no fully self-hostable deployment. To re-enable, use their hosted "
            "SaaS and restore the registry entry in src/memory/__init__.py."
        )

    def _connect(self) -> Any:  # pragma: no cover - unreachable
        raise NotImplementedError
