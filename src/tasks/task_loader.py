"""Load DFx tasks and rule packs from disk."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable

from ..utils import get_logger, load_config
from .dfx_task import DFxTask, RulePack

logger = get_logger(__name__)


class TaskLoader:
    """Discovers task and rule files on disk and parses them into models."""

    SUPPORTED_SUFFIXES = (".yaml", ".yml", ".json")

    def __init__(self, tasks_dir: str | Path) -> None:
        self.tasks_dir = Path(tasks_dir)
        if not self.tasks_dir.exists():
            raise FileNotFoundError(f"Tasks directory does not exist: {self.tasks_dir}")

    def load_all(self) -> list[DFxTask]:
        """Discover and parse every task file under ``tasks_dir`` (recursive)."""
        tasks: list[DFxTask] = []
        for p in sorted(self.tasks_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in self.SUPPORTED_SUFFIXES:
                try:
                    tasks.append(self.load_task(p))
                except Exception as exc:  # noqa: BLE001 - surface, don't crash
                    logger.warning("Skipping malformed task file %s: %s", p, exc)
        logger.info("Loaded %d DFx tasks from %s", len(tasks), self.tasks_dir)
        return tasks

    def load_task(self, path: str | Path) -> DFxTask:
        """Parse a single task file into a ``DFxTask``."""
        raw = load_config(path)
        return DFxTask.model_validate(raw)

    @staticmethod
    def filter_tasks(
        tasks: Iterable[DFxTask],
        include_scenarios: list[str] | None = None,
        exclude_scenarios: list[str] | None = None,
    ) -> list[DFxTask]:
        include = set(include_scenarios or [])
        exclude = set(exclude_scenarios or [])
        out = []
        for t in tasks:
            if include and t.scenario_name not in include:
                continue
            if t.scenario_name in exclude:
                continue
            out.append(t)
        return out


@lru_cache(maxsize=64)
def load_rule_pack(path: str) -> RulePack:
    """Cached rule-pack loader (same path is read once per process)."""
    raw = load_config(path)
    return RulePack.model_validate(raw)
