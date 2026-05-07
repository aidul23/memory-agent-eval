"""Append-only JSONL writer with robust serialization.

The experiment runner writes one JSON object per line per agent interaction.
Using JSONL (vs a single big JSON array) means partial runs remain readable
and crashes never corrupt the file.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _default_serializer(obj: Any) -> Any:
    """Best-effort JSON serializer for non-standard types."""
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    return repr(obj)


class JsonlWriter:
    """Thread-safe append-only writer.

    Each ``write`` call flushes immediately to keep partial results durable.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, record: dict[str, Any]) -> None:
        record = {**record}
        record.setdefault("logged_at", datetime.now(timezone.utc).isoformat())
        line = json.dumps(record, default=_default_serializer, ensure_ascii=False)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
                fh.flush()
                os.fsync(fh.fileno())

    def write_many(self, records: list[dict[str, Any]]) -> None:
        for r in records:
            self.write(r)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"JsonlWriter(path={self.path!s})"
