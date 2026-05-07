"""Memory subsystem.

Each concrete memory class implements ``BaseMemory`` (retrieve / update /
reset / export_memory). The factory below maps config strings to classes
so the experiment runner can instantiate them by name.
"""

from __future__ import annotations

from typing import Any

from .acontext_memory import AContextMemory
from .base_memory import BaseMemory, MemoryItem
from .contextual_memory import ContextualMemory
from .hindsight_memory import HindsightMemory
from .mem0_memory import Mem0Memory
from .persistent_memory import PersistentMemory
from .stateless_memory import StatelessMemory
from .supermemory_memory import SupermemoryMemory
from .zep_memory import ZepMemory

_REGISTRY: dict[str, type[BaseMemory]] = {
    "stateless": StatelessMemory,
    "hindsight": HindsightMemory,
    "contextual": ContextualMemory,
    "persistent": PersistentMemory,
    "mem0": Mem0Memory,
    "zep": ZepMemory,
    "supermemory": SupermemoryMemory,
    "acontext": AContextMemory,
}


def create_memory(name: str, **kwargs: Any) -> BaseMemory:
    """Instantiate a memory by registry name. Extra kwargs forwarded to ctor."""
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown memory system {name!r}. Known: {sorted(_REGISTRY)}"
        )
    cls = _REGISTRY[key]
    # Drop the optional ``class`` key (used in YAML for documentation).
    kwargs.pop("class", None)
    return cls(**kwargs)


__all__ = [
    "AContextMemory",
    "BaseMemory",
    "ContextualMemory",
    "HindsightMemory",
    "Mem0Memory",
    "MemoryItem",
    "PersistentMemory",
    "StatelessMemory",
    "SupermemoryMemory",
    "ZepMemory",
    "create_memory",
]
