"""Provider interface. Generation nodes talk only to this — never to a vendor SDK
directly — so mock and Claude are perfectly swappable."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    meta: dict = field(default_factory=dict)


class LLMProvider(ABC):
    name: str = "base"

    @abstractmethod
    def complete(self, system: str, user: str, **kwargs) -> LLMResponse:
        """Free-form completion (used by generation nodes)."""

    @abstractmethod
    def judge(self, system: str, user: str, **kwargs) -> dict:
        """Structured rubric score: returns {"score": int 1-5, "rationale": str}."""
