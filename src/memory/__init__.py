"""Memory subsystem.

Each concrete memory class implements ``BaseMemory`` (retrieve / update /
reset / export_memory). The factory below maps config strings to classes
so the experiment runner can instantiate them by name.

Registry layout
---------------
- ``stateless`` - no-op baseline
- ``reflection`` - our hand-rolled feedback reflections (Jaccard retrieval)
- ``contextual`` - richer per-task context store (Jaccard retrieval)
- ``persistent`` - JSONL-backed long-term store
- ``mem0`` - Mem0 (self-hosted via Qdrant + LLM)
- ``zep`` - Zep Community Edition (self-hosted via Docker)
- ``acontext`` - AContext (self-hosted via ``acontext server up``)
- ``hindsight`` - Vectorize.io Hindsight (self-hosted via Docker)

Backwards compatibility
-----------------------
The historical ``hindsight`` key (which used to point at our internal
reflection technique) now refers to Vectorize.io's Hindsight service. The
internal technique was renamed to ``ReflectionMemory`` and registered
under ``reflection``. Old configs using ``hindsight`` will silently load
the Vectorize wrapper; if you want the old behaviour, switch to
``reflection``.

Supermemory is intentionally NOT registered because Supermemory has no
fully self-hostable deployment yet. To re-enable, see
``src/memory/supermemory_memory.py``.
"""

from __future__ import annotations

from typing import Any

from .acontext_memory import AContextMemory
from .base_memory import BaseMemory, MemoryItem
from .contextual_memory import ContextualMemory
from .mem0_memory import Mem0Memory
from .persistent_memory import PersistentMemory
from .reflection_memory import ReflectionMemory
from .stateless_memory import StatelessMemory
from .vectorize_hindsight_memory import VectorizeHindsightMemory
from .zep_memory import ZepMemory

_REGISTRY: dict[str, type[BaseMemory]] = {
    "stateless": StatelessMemory,
    "reflection": ReflectionMemory,
    "contextual": ContextualMemory,
    "persistent": PersistentMemory,
    "mem0": Mem0Memory,
    "zep": ZepMemory,
    "acontext": AContextMemory,
    "hindsight": VectorizeHindsightMemory,
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
    "Mem0Memory",
    "MemoryItem",
    "PersistentMemory",
    "ReflectionMemory",
    "StatelessMemory",
    "VectorizeHindsightMemory",
    "ZepMemory",
    "create_memory",
]
