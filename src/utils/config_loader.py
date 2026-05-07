"""YAML / JSON config loading helpers.

Keeps config parsing in one place so the rest of the codebase can stay free of
file I/O concerns. ``merge_configs`` performs a deep recursive merge with
``override`` taking precedence, used to layer experiment.yaml on top of a
default base config.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML or JSON file and return a plain dict.

    Falls back to YAML parsing if the suffix is unknown - YAML is a superset
    of JSON so this is safe.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")

    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        return json.loads(text)
    return yaml.safe_load(text) or {}


def merge_configs(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Deep-merge two mappings. ``override`` wins on conflicts.

    Lists are replaced wholesale (not concatenated) - this matches user
    expectations when overriding e.g. ``memory_systems`` in experiment configs.
    """
    out: dict[str, Any] = dict(base)
    for key, val in override.items():
        if (
            key in out
            and isinstance(out[key], Mapping)
            and isinstance(val, Mapping)
        ):
            out[key] = merge_configs(out[key], val)
        else:
            out[key] = val
    return out
