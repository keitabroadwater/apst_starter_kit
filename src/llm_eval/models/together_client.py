"""Together.ai model runner."""

from __future__ import annotations

import logging
import time
from typing import Any

from llm_eval.generation.rate_limiter import RateLimitedExecutor
from llm_eval.models.base import GenerationOutput, ModelRunner
from llm_eval.models.catalog import expand_together_model_name

logger = logging.getLogger(__name__)


class TogetherModelRunner(ModelRunner):
    """Together.ai chat completions runner."""

    def __init__(
        self,
        model_name: str,
        *,
        api_key: str | None = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        max_concurrent_requests: int = 5,
        max_requests_per_second: float = 2.0,
    ) -> None:
        super().__init__(expand_together_model_name(model_name))
        try:
            from together import Together
        except ImportError as exc:
            raise RuntimeError("Install Together.ai support with: pip install -e '.[providers]'") from exc

        self.client = Together(api_key=api_key)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.rate_limiter = RateLimitedExecutor(
            max_concurrent_requests=max_concurrent_requests,
            max_requests_per_second=max_requests_per_second,
        )

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
                response = self.rate_limiter.execute(
                    self.client.chat.completions.create,
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                )
                choice = response.choices[0]
                usage = getattr(response, "usage", None)
                return GenerationOutput(
                    text=(choice.message.content or "").strip(),
                    model_name=self.model_name,
                    input_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
                    output_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
                    finish_reason=str(getattr(choice, "finish_reason", "") or "") or None,
                    raw=response,
                )
            except Exception as exc:  # noqa: BLE001 - provider SDK exceptions vary.
                message = str(exc).lower()
                if "dedicated_endpoint" in message or "dedicated endpoint" in message:
                    return GenerationOutput(
                        text="",
                        model_name=self.model_name,
                        error=f"Dedicated endpoint required for {self.model_name}: {exc}",
                    )
                if attempt + 1 >= self.max_retries:
                    return GenerationOutput(
                        text="",
                        model_name=self.model_name,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                wait = _retry_wait(self.retry_delay, attempt, message)
                logger.warning("Together generation failed; retrying in %.1fs: %s", wait, exc)
                time.sleep(wait)

        return GenerationOutput(text="", model_name=self.model_name, error="unknown generation error")


def _retry_wait(base_delay: float, attempt: int, message: str) -> float:
    if "429" in message or "rate limit" in message or "too many requests" in message:
        return max(5.0, base_delay * (2 ** (attempt + 2)))
    return base_delay * (2**attempt)

