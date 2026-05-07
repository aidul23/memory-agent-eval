"""Google Gemini provider via google-generativeai."""

from __future__ import annotations

import os
import time
from typing import Any

from ..utils import get_logger
from .base_llm import BaseLLM, LLMResponse

logger = get_logger(__name__)

_DEFAULT_INPUT_COSTS = {
    "gemini-1.5-pro": 0.00125,
    "gemini-1.5-flash": 0.000075,
}
_DEFAULT_OUTPUT_COSTS = {
    "gemini-1.5-pro": 0.005,
    "gemini-1.5-flash": 0.0003,
}


class GoogleLLM(BaseLLM):
    provider = "google"

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
            import google.generativeai as genai
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "google-generativeai is required for GoogleLLM."
            ) from exc

        genai.configure(api_key=api_key or os.getenv("GOOGLE_API_KEY"))
        self._genai = genai
        self._gen_model = genai.GenerativeModel(model)
        self._timeout = request_timeout
        self._in_cost = cost_per_1k_input_tokens or _DEFAULT_INPUT_COSTS
        self._out_cost = cost_per_1k_output_tokens or _DEFAULT_OUTPUT_COSTS

    def generate(self, messages: list[dict[str, str]], temperature: float = 0.0) -> LLMResponse:
        system_parts: list[str] = []
        history: list[dict[str, Any]] = []
        for m in messages:
            role = m.get("role")
            content = m.get("content", "")
            if role == "system":
                system_parts.append(content)
            elif role in ("user", "assistant"):
                gem_role = "user" if role == "user" else "model"
                history.append({"role": gem_role, "parts": [content]})

        if system_parts and history:
            history[0]["parts"][0] = "[system]\n" + "\n".join(system_parts) + "\n\n" + history[0]["parts"][0]

        t0 = time.perf_counter()
        resp = self._gen_model.generate_content(
            history,
            generation_config={"temperature": temperature},
            request_options={"timeout": self._timeout},
        )
        latency = time.perf_counter() - t0

        text = getattr(resp, "text", "") or ""
        usage = getattr(resp, "usage_metadata", None)
        in_tok = int(getattr(usage, "prompt_token_count", 0) or 0)
        out_tok = int(getattr(usage, "candidates_token_count", 0) or 0)

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
            finish_reason="stop",
            raw={"provider": self.provider, "model": self.model},
        )
