"""OpenAI model runner."""

from __future__ import annotations

import logging
import time
from typing import Any

from llm_eval.models.base import GenerationOutput, ModelRunner

logger = logging.getLogger(__name__)


class OpenAIModelRunner(ModelRunner):
    """OpenAI chat completions runner."""

    def __init__(
        self,
        model_name: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        super().__init__(model_name)
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install OpenAI support with: pip install -e '.[providers]'") from exc

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 1.0,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> GenerationOutput:
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                )
                choice = response.choices[0]
                usage = getattr(response, "usage", None)
                return GenerationOutput(
                    text=_extract_text(choice.message.content),
                    model_name=self.model_name,
                    input_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
                    output_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
                    finish_reason=str(getattr(choice, "finish_reason", "") or "") or None,
                    raw=response,
                )
            except Exception as exc:  # noqa: BLE001 - provider SDK exceptions vary.
                if attempt + 1 >= self.max_retries:
                    return GenerationOutput(
                        text="",
                        model_name=self.model_name,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                wait = self.retry_delay * (2**attempt)
                logger.warning("OpenAI generation failed; retrying in %.1fs: %s", wait, exc)
                time.sleep(wait)

        return GenerationOutput(text="", model_name=self.model_name, error="unknown generation error")


def _extract_text(content: Any) -> str:
    """Handle plain and structured OpenAI content."""

    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part.strip() for part in parts if part and part.strip())
    return str(content).strip()

