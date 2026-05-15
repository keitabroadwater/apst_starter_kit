"""Model runner factory."""

from __future__ import annotations

from llm_eval.config import ModelConfig
from llm_eval.models.base import ModelRunner
from llm_eval.models.catalog import looks_like_together_model
from llm_eval.models.mock_client import MockModelRunner
from llm_eval.models.openai_client import OpenAIModelRunner
from llm_eval.models.together_client import TogetherModelRunner

OPENAI_COMPATIBLE_PROVIDERS = {
    "openai_compatible",
    "openai-compatible",
    "local",
    "ollama",
    "vllm",
}


def infer_provider(model_id: str) -> str:
    if model_id.lower().startswith("mock"):
        return "mock"
    return "together" if looks_like_together_model(model_id) else "openai"


def create_model_runner(
    config: ModelConfig,
    *,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    max_concurrent_requests: int = 5,
    max_requests_per_second: float = 2.0,
) -> ModelRunner:
    """Create a provider runner from config."""

    provider = config.provider.lower()
    if provider in {"mock", "demo"}:
        return MockModelRunner(config.model_id)
    if provider in OPENAI_COMPATIBLE_PROVIDERS and not config.base_url:
        raise ValueError(f"Provider {config.provider!r} requires base_url")
    if provider in OPENAI_COMPATIBLE_PROVIDERS:
        return OpenAIModelRunner(
            config.model_id,
            api_key=config.api_key,
            base_url=config.base_url,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
    if provider == "openai":
        return OpenAIModelRunner(
            config.model_id,
            api_key=config.api_key,
            base_url=config.base_url,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
    if provider == "together":
        return TogetherModelRunner(
            config.model_id,
            api_key=config.api_key,
            max_retries=max_retries,
            retry_delay=retry_delay,
            max_concurrent_requests=max_concurrent_requests,
            max_requests_per_second=max_requests_per_second,
        )
    raise ValueError(f"Unsupported provider: {config.provider}")
