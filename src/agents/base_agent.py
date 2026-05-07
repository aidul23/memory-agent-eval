"""Base agent contract.

A concrete agent receives a task, returns a parsed structured response, and
exposes the prompt + raw text + LLM response so the runner can log every
intermediate artefact.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

from ..llms.base_llm import LLMResponse
from ..memory.base_memory import MemoryItem
from ..tasks.dfx_task import DFxTask


@dataclass
class AgentResult:
    """Everything produced by one agent run on one task."""

    response: dict[str, Any]
    raw_text: str
    prompt_messages: list[dict[str, str]]
    retrieved_memory: list[MemoryItem] = field(default_factory=list)
    llm_response: LLMResponse | None = None

    def memory_as_dicts(self) -> list[dict[str, Any]]:
        return [m.to_dict() for m in self.retrieved_memory]


class BaseAgent(abc.ABC):
    """Agent contract used by the experiment runner."""

    @abc.abstractmethod
    def run(self, task: DFxTask, prior_feedback: dict[str, Any] | None = None) -> AgentResult:
        """Execute one task and return a structured AgentResult."""
