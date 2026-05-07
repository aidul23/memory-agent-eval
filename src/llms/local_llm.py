"""Local model provider via Ollama HTTP API.

Ollama exposes an OpenAI-compatible /api/chat endpoint - we use plain
``requests`` to keep this dependency-light. Tokenisation reported by Ollama
is approximate so cost estimation is left at zero by default.
"""

from __future__ import annotations

import os
import time
from typing import Any

import requests

from ..utils import get_logger
from .base_llm import BaseLLM, LLMResponse

logger = get_logger(__name__)


class LocalLLM(BaseLLM):
    provider = "local"

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        request_timeout: float = 120.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self.base_url = (
            base_url
            or os.getenv("OLLAMA_BASE_URL")
            or "http://localhost:11434"
        ).rstrip("/")
        self._timeout = request_timeout

    def generate(self, messages: list[dict[str, str]], temperature: float = 0.0) -> LLMResponse:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        t0 = time.perf_counter()
        resp = requests.post(url, json=payload, timeout=self._timeout)
        resp.raise_for_status()
        data = resp.json()
        latency = time.perf_counter() - t0

        msg = data.get("message", {}) or {}
        text = msg.get("content", "") or ""
        in_tok = int(data.get("prompt_eval_count", 0) or 0)
        out_tok = int(data.get("eval_count", 0) or 0)

        return LLMResponse(
            text=text,
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_s=latency,
            estimated_cost_usd=0.0,
            finish_reason=data.get("done_reason"),
            raw={"provider": self.provider, "model": self.model, "endpoint": url},
        )
