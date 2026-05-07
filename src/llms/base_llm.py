"""Abstract LLM interface.

All providers implement ``generate`` and return an ``LLMResponse`` with the
text content plus enough metadata for the metrics module to compute
latency, token usage, and cost.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMResponse:
    """Provider-agnostic LLM call result."""

    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    estimated_cost_usd: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)
    finish_reason: str | None = None


class BaseLLM(abc.ABC):
    """Provider-agnostic LLM contract."""

    provider: str = "base"

    def __init__(self, model: str, **kwargs: Any) -> None:
        self.model = model
        self.kwargs = kwargs

    @abc.abstractmethod
    def generate(self, messages: list[dict[str, str]], temperature: float = 0.0) -> LLMResponse:
        """Generate a completion.

        ``messages`` follows the OpenAI chat schema: a list of dicts with
        ``role`` (system | user | assistant) and ``content`` keys. Providers
        translate to their native format internally.
        """
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.__class__.__name__}(model={self.model!r})"
