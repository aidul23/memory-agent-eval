"""OpenAI provider.

Uses the official ``openai`` SDK (>=1.x). The class accepts standard chat
messages and returns a fully-populated ``LLMResponse`` with token usage and
estimated cost.
"""

from __future__ import annotations

import os
import time
from typing import Any

from ..utils import get_logger
from .base_llm import BaseLLM, LLMResponse

logger = get_logger(__name__)

# Default per-1k token costs - overridden by configs/models.yaml when loaded
# via the experiment runner.
_DEFAULT_INPUT_COSTS = {
    "gpt-4o-mini": 0.00015,
    "gpt-4o": 0.0025,
    "gpt-4-turbo": 0.01,
    "o3": 0.015,
}
_DEFAULT_OUTPUT_COSTS = {
    "gpt-4o-mini": 0.0006,
    "gpt-4o": 0.01,
    "gpt-4-turbo": 0.03,
    "o3": 0.06,
}


class OpenAILLM(BaseLLM):
    provider = "openai"

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        cost_per_1k_input_tokens: dict[str, float] | None = None,
        cost_per_1k_output_tokens: dict[str, float] | None = None,
        request_timeout: float = 60.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "openai>=1.0 is required for OpenAILLM. `pip install openai`."
            ) from exc

        self._client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            timeout=request_timeout,
        )
        self._in_cost = cost_per_1k_input_tokens or _DEFAULT_INPUT_COSTS
        self._out_cost = cost_per_1k_output_tokens or _DEFAULT_OUTPUT_COSTS

    def generate(self, messages: list[dict[str, str]], temperature: float = 0.0) -> LLMResponse:
        t0 = time.perf_counter()
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        latency = time.perf_counter() - t0

        choice = resp.choices[0]
        usage = getattr(resp, "usage", None)
        in_tok = int(getattr(usage, "prompt_tokens", 0) or 0)
        out_tok = int(getattr(usage, "completion_tokens", 0) or 0)

        cost = (
            in_tok / 1000.0 * self._in_cost.get(self.model, 0.0)
            + out_tok / 1000.0 * self._out_cost.get(self.model, 0.0)
        )

        return LLMResponse(
            text=choice.message.content or "",
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_s=latency,
            estimated_cost_usd=cost,
            finish_reason=getattr(choice, "finish_reason", None),
            raw={"provider": self.provider, "model": self.model, "id": getattr(resp, "id", None)},
        )
