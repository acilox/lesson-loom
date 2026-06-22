"""Provider factory: mock unless Claude is explicitly requested AND a key exists."""

from __future__ import annotations

import os

from lesson_loom.providers.base import LLMProvider, LLMResponse
from lesson_loom.providers.mock import MockProvider

__all__ = ["LLMProvider", "LLMResponse", "MockProvider", "build_provider"]


def build_provider(
    provider_name: str,
    generation_model: str = "claude-sonnet-4-6",
    judge_model: str = "claude-opus-4-8",
    seed: int = 42,
) -> LLMProvider:
    if provider_name == "claude" and os.getenv("ANTHROPIC_API_KEY"):
        from lesson_loom.providers.claude import ClaudeProvider

        return ClaudeProvider(generation_model=generation_model, judge_model=judge_model)
    # Default everywhere else: fully offline, deterministic.
    return MockProvider(seed=seed)
