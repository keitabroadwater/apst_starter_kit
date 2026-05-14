"""Base model runner protocol."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GenerationOutput:
    """One model generation plus metadata."""

    text: str
    model_name: str
    input_tokens: int = 0
    output_tokens: int = 0
    finish_reason: str | None = None
    error: str | None = None
    raw: Any | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class ModelRunner(ABC):
    """Abstract interface implemented by provider clients."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 1.0,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> GenerationOutput:
        """Generate one response."""

    def generate_text(
        self,
        prompt: str,
        *,
        temperature: float = 1.0,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate text and raise if the provider call fails."""

        output = self.generate(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        if output.error:
            raise RuntimeError(output.error)
        return output.text

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_name={self.model_name!r})"

