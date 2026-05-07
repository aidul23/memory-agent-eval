"""Anthropic Claude provider."""

from __future__ import annotations

import os
import time
from typing import Any

from ..utils import get_logger
from .base_llm import BaseLLM, LLMResponse

logger = get_logger(__name__)

_DEFAULT_INPUT_COSTS = {
    "claude-3-5-sonnet-latest": 0.003,
    "claude-3-5-haiku-latest": 0.0008,
    "claude-opus-4": 0.015,
}
_DEFAULT_OUTPUT_COSTS = {
    "claude-3-5-sonnet-latest": 0.015,
    "claude-3-5-haiku-latest": 0.004,
    "claude-opus-4": 0.075,
}


class AnthropicLLM(BaseLLM):
    provider = "anthropic"

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        cost_per_1k_input_tokens: dict[str, float] | None = None,
        cost_per_1k_output_tokens: dict[str, float] | None = None,
        request_timeout: float = 60.0,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "anthropic SDK is required for AnthropicLLM. `pip install anthropic`."
            ) from exc

        self._client = anthropic.Anthropic(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY"),
            timeout=request_timeout,
        )
        self._in_cost = cost_per_1k_input_tokens or _DEFAULT_INPUT_COSTS
        self._out_cost = cost_per_1k_output_tokens or _DEFAULT_OUTPUT_COSTS
        self._max_tokens = max_tokens

    def generate(self, messages: list[dict[str, str]], temperature: float = 0.0) -> LLMResponse:
        # Anthropic separates the system prompt from the user/assistant chain.
        system_parts: list[str] = []
        chat: list[dict[str, str]] = []
        for m in messages:
            if m.get("role") == "system":
                system_parts.append(m.get("content", ""))
            else:
                chat.append({"role": m["role"], "content": m.get("content", "")})

        t0 = time.perf_counter()
        resp = self._client.messages.create(
            model=self.model,
            system="\n\n".join(system_parts) if system_parts else None,
            messages=chat,
            temperature=temperature,
            max_tokens=self._max_tokens,
        )
        latency = time.perf_counter() - t0

        text = ""
        for block in getattr(resp, "content", []) or []:
            if getattr(block, "type", None) == "text":
                text += getattr(block, "text", "")

        usage = getattr(resp, "usage", None)
        in_tok = int(getattr(usage, "input_tokens", 0) or 0)
        out_tok = int(getattr(usage, "output_tokens", 0) or 0)
        cost = (
            in_tok / 1000.0 * self._in_cost.get(self.model, 0.0)
            + out_tok / 1000.0 * self._out_cost.get(self.model, 0.0)
        )

        return LLMResponse(
            text=text,
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_s=latency,
            estimated_cost_usd=cost,
            finish_reason=getattr(resp, "stop_reason", None),
            raw={"provider": self.provider, "model": self.model, "id": getattr(resp, "id", None)},
        )
