"""Simple cost and runtime estimators.

Provider prices change. Treat these defaults as planning placeholders and
update them before making budget decisions.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPricing:
    provider: str
    input_price_per_1m: float
    output_price_per_1m: float


PRICING: dict[str, ModelPricing] = {
    "gpt-4o-mini": ModelPricing("openai", 0.15, 0.60),
    "gpt-4o": ModelPricing("openai", 2.50, 10.00),
    "gpt-4.1": ModelPricing("openai", 2.00, 8.00),
    "openai/gpt-oss-20b": ModelPricing("together", 0.05, 0.20),
    "google/gemma-2b-it": ModelPricing("together", 0.01, 0.02),
    "Qwen/Qwen2.5-7B-Instruct-Turbo": ModelPricing("together", 0.05, 0.15),
    "meta-llama/Llama-3.1-8B-Instruct": ModelPricing("together", 0.10, 0.10),
}


def estimate_cost(
    *,
    model: str,
    n_calls: int,
    input_tokens: int = 600,
    output_tokens: int = 300,
) -> dict[str, float | str | bool]:
    pricing = PRICING.get(model)
    if pricing is None:
        return {"model": model, "pricing_available": False, "estimated_cost": 0.0}

    per_call = (
        input_tokens * pricing.input_price_per_1m / 1_000_000
        + output_tokens * pricing.output_price_per_1m / 1_000_000
    )
    return {
        "model": model,
        "provider": pricing.provider,
        "pricing_available": True,
        "cost_per_call": per_call,
        "estimated_cost": per_call * n_calls,
    }


def estimate_experiment_cost(
    *,
    models: list[str],
    n_prompts: int,
    n_temperatures: int,
    n_samples: int,
    judge_model: str = "gpt-4o-mini",
) -> dict[str, object]:
    generation_calls = n_prompts * n_temperatures * n_samples
    model_costs = [
        estimate_cost(model=model, n_calls=generation_calls)
        for model in models
    ]
    judge_calls = generation_calls * len(models)
    judge_cost = estimate_cost(model=judge_model, n_calls=judge_calls, input_tokens=900, output_tokens=100)
    total = sum(float(item["estimated_cost"]) for item in model_costs) + float(
        judge_cost["estimated_cost"]
    )
    return {
        "generation_calls_per_model": generation_calls,
        "judge_calls": judge_calls,
        "models": model_costs,
        "judge": judge_cost,
        "estimated_total_cost": total,
    }

