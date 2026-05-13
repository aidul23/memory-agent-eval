"""Backwards-compatibility shim.

``HindsightMemory`` was renamed to :class:`ReflectionMemory` to free up the
``hindsight`` registry slot for Vectorize.io's Hindsight memory service
(see :mod:`vectorize_hindsight_memory`).

Importing ``HindsightMemory`` from this module still works but emits a
``DeprecationWarning``. New code should import ``ReflectionMemory`` from
``src.memory`` or ``src.memory.reflection_memory``.
"""

from __future__ import annotations

import warnings
from typing import Any

from .reflection_memory import ReflectionMemory


class HindsightMemory(ReflectionMemory):
    """Deprecated alias for :class:`ReflectionMemory`."""

    name = "reflection"  # keep behaviour identical; new registry key is 'reflection'

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        warnings.warn(
            "HindsightMemory has been renamed to ReflectionMemory and the "
            "registry key changed from 'hindsight' to 'reflection'. The "
            "'hindsight' key now refers to Vectorize.io's Hindsight memory "
            "service (VectorizeHindsightMemory). Update your imports and "
            "configs.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


__all__ = ["HindsightMemory"]
