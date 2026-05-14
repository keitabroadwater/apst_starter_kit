"""Small model catalog helpers."""

from __future__ import annotations


TOGETHER_ON_DEMAND_MODELS: dict[str, str] = {
    "gemma-2b": "google/gemma-2b-it",
    "gemma-3n-e4b": "google/gemma-3n-E4B-it",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct-Turbo",
    "gpt-oss-20b": "openai/gpt-oss-20b",
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama-3.2-3b-turbo": "meta-llama/Llama-3.2-3B-Instruct-Turbo",
    "mistral-small-24b": "mistralai/Mistral-Small-24B-Instruct",
    "mistral-small": "mistralai/Mistral-Small-24B-Instruct-25.01",
}

TOGETHER_DEDICATED_ENDPOINT_MODELS: set[str] = {
    "google/gemma-2-9b-it",
    "google/gemma-2-27b-it",
    "nvidia/Nemotron-Nano-9B-V2",
    "Qwen/Qwen3-235B-FP8-Throughput",
    "openai/gpt-oss-120b",
    "meta-llama/Llama-4-Scout-Instruct",
    "meta-llama/Llama-4-Maverick-Instruct",
}


def expand_together_model_name(name: str) -> str:
    """Return a full Together.ai model ID for known short names."""

    model_id = TOGETHER_ON_DEMAND_MODELS.get(name.lower(), name)
    if model_id in TOGETHER_DEDICATED_ENDPOINT_MODELS:
        raise ValueError(
            f"{model_id} is listed as requiring a dedicated endpoint. "
            "Use an on-demand Together.ai model or update the catalog after validation."
        )
    return model_id


def looks_like_together_model(model_id: str) -> bool:
    """Infer whether a model ID belongs to Together.ai."""

    if model_id in TOGETHER_ON_DEMAND_MODELS:
        return True
    if "/" in model_id:
        return True
    return model_id.startswith(("meta-llama", "google", "mistralai", "Qwen", "nvidia"))

