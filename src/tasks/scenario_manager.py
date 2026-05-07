"""Group tasks into ordered scenarios.

A scenario is a sequence of related sessions sharing a ``scenario_name``.
The runner iterates scenarios so that memory-dependent agents see the sessions
in the correct order.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

from .dfx_task import DFxTask


@dataclass(frozen=True)
class Scenario:
    name: str
    sessions: list[DFxTask] = field(default_factory=list)

    def __iter__(self):
        return iter(self.sessions)

    def __len__(self) -> int:
        return len(self.sessions)


class ScenarioManager:
    """Group ``DFxTask`` objects by scenario_name and sort by session_id."""

    def __init__(self, tasks: Iterable[DFxTask]) -> None:
        bucket: dict[str, list[DFxTask]] = defaultdict(list)
        for t in tasks:
            bucket[t.scenario_name].append(t)
        self._scenarios: list[Scenario] = []
        for name, sessions in bucket.items():
            sessions_sorted = sorted(sessions, key=lambda s: s.session_id)
            self._scenarios.append(Scenario(name=name, sessions=sessions_sorted))
        self._scenarios.sort(key=lambda s: s.name)

    @property
    def scenarios(self) -> list[Scenario]:
        return list(self._scenarios)

    def __iter__(self):
        return iter(self._scenarios)

    def __len__(self) -> int:
        return len(self._scenarios)
