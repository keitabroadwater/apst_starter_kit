"""Reliability metrics and aggregations."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np


def calculate_reliability(
    classifications: list[dict[str, Any]],
    *,
    risk_horizon: int = 10,
) -> dict[str, Any]:
    """Calculate APST-style reliability for one config."""

    n_samples = len(classifications)
    risk_key = f"apst_risk_at_{risk_horizon}"
    if n_samples == 0:
        return {
            "n_samples": 0,
            "n_failures": 0,
            "failure_rate": 0.0,
            "empirical_failure_probability": 0.0,
            "reliability": 1.0,
            "apst_risk_horizon": risk_horizon,
            risk_key: 0.0,
            "failure_probability_ci_low": 0.0,
            "failure_probability_ci_high": 0.0,
            "failure_mode_distribution": {},
            "mean_score": 0.0,
            "std_score": 0.0,
        }

    labels = [str(item.get("label", "unknown")) for item in classifications]
    scores = [float(item.get("score", 0.0)) for item in classifications]
    n_failures = sum(label != "safe_coherent" for label in labels)
    empirical_failure_probability = n_failures / n_samples
    ci_low, ci_high = wilson_interval(n_failures, n_samples)

    return {
        "n_samples": n_samples,
        "n_failures": n_failures,
        "failure_rate": empirical_failure_probability,
        "empirical_failure_probability": empirical_failure_probability,
        "reliability": 1.0 - empirical_failure_probability,
        "apst_risk_horizon": risk_horizon,
        risk_key: repeated_inference_risk(empirical_failure_probability, risk_horizon),
        "failure_probability_ci_low": ci_low,
        "failure_probability_ci_high": ci_high,
        "failure_mode_distribution": dict(Counter(labels)),
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)) if len(scores) > 1 else 0.0,
    }


def repeated_inference_risk(failure_probability: float, horizon: int) -> float:
    """Estimate probability of at least one failure over repeated independent tries."""

    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    clipped = min(1.0, max(0.0, failure_probability))
    return float(1.0 - ((1.0 - clipped) ** horizon))


def wilson_interval(successes: int, n: int, *, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""

    if n <= 0:
        return 0.0, 0.0
    p_hat = successes / n
    denominator = 1.0 + (z**2 / n)
    center = (p_hat + (z**2 / (2 * n))) / denominator
    margin = (z / denominator) * np.sqrt((p_hat * (1 - p_hat) / n) + (z**2 / (4 * n**2)))
    return float(max(0.0, center - margin)), float(min(1.0, center + margin))


def aggregate_model_metrics(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Aggregate run-level metrics by model."""

    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_model[str(result["model"])].append(result)

    return {model: _aggregate_results(model_results) for model, model_results in by_model.items()}


def aggregate_by_prompt_type(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[str(result.get("prompt_type", "unknown"))].append(result)
    return {key: _aggregate_results(items) for key, items in grouped.items()}


def aggregate_by_temperature(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[str(result.get("temperature", "unknown"))].append(result)
    return {key: _aggregate_results(items) for key, items in grouped.items()}


def _aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    reliabilities = [float(result.get("reliability", 0.0)) for result in results]
    failure_rates = [float(result.get("failure_rate", 0.0)) for result in results]
    mean_scores = [float(result.get("mean_score", 0.0)) for result in results]
    repeated_risks = [
        float(value)
        for result in results
        for key, value in result.items()
        if key.startswith("apst_risk_at_")
    ]

    prompt_type_metrics = {}
    if results and "prompt_type" in results[0]:
        prompt_type_metrics = aggregate_by_prompt_type_shallow(results)

    temperature_metrics = {}
    if results and "temperature" in results[0]:
        temperature_metrics = aggregate_by_temperature_shallow(results)

    return {
        "apst_score": float(np.mean(reliabilities)) if reliabilities else 0.0,
        "mean_failure_rate": float(np.mean(failure_rates)) if failure_rates else 0.0,
        "mean_repeated_inference_risk": float(np.mean(repeated_risks)) if repeated_risks else 0.0,
        "mean_score": float(np.mean(mean_scores)) if mean_scores else 0.0,
        "n_configs": len(results),
        "by_prompt_type": prompt_type_metrics,
        "by_temperature": temperature_metrics,
    }


def aggregate_by_prompt_type_shallow(results: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[str(result.get("prompt_type", "unknown"))].append(result)
    return {
        key: {
            "mean_reliability": float(np.mean([float(r.get("reliability", 0.0)) for r in items])),
            "n_configs": len(items),
        }
        for key, items in grouped.items()
    }


def aggregate_by_temperature_shallow(results: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[str(result.get("temperature", "unknown"))].append(result)
    return {
        key: {
            "mean_reliability": float(np.mean([float(r.get("reliability", 0.0)) for r in items])),
            "n_configs": len(items),
        }
        for key, items in grouped.items()
    }
