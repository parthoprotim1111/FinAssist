from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class GenerationResult:
    text: str
    raw: dict[str, Any] | None = None


class LLMBackend(ABC):
    """Abstract interface for all backends (HF primary, HF alt, mock)."""

    name: str = "base"

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int | None = None,
        temperature: float = 0.2,
        max_input_tokens: int | None = None,
        truncation_side: str | None = None,
    ) -> GenerationResult:
        raise NotImplementedError

    def unload(self) -> None:
        """Free GPU memory when swapping models."""
        pass
