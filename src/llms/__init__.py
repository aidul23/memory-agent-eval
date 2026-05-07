"""LLM provider abstraction.

Importing concrete providers is deferred so the platform stays usable even
when optional SDKs (openai, anthropic, google-generativeai) are not installed.
"""

from __future__ import annotations

from typing import Any

from .base_llm import BaseLLM, LLMResponse


def create_llm(provider: str, model: str, **kwargs: Any) -> BaseLLM:
    """Factory: instantiate a concrete LLM by string identifier.

    Falls back to a clear ImportError-derived error if the requested SDK
    isn't installed.
    """
    provider = provider.lower()
    if provider == "openai":
        from .openai_llm import OpenAILLM

        return OpenAILLM(model=model, **kwargs)
    if provider in {"google", "gemini"}:
        from .google_llm import GoogleLLM

        return GoogleLLM(model=model, **kwargs)
    if provider in {"anthropic", "claude"}:
        from .anthropic_llm import AnthropicLLM

        return AnthropicLLM(model=model, **kwargs)
    if provider in {"local", "ollama"}:
        from .local_llm import LocalLLM

        return LocalLLM(model=model, **kwargs)
    if provider in {"mock", "stub", "fake"}:
        from .mock_llm import MockLLM

        return MockLLM(model=model, **kwargs)
    raise ValueError(f"Unknown LLM provider: {provider!r}")


__all__ = ["BaseLLM", "LLMResponse", "create_llm"]
