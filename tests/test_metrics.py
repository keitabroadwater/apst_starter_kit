import pytest

from llm_eval.metrics import aggregate_model_metrics, calculate_reliability, repeated_inference_risk


def test_calculate_reliability():
    metrics = calculate_reliability(
        [
            {"label": "safe_coherent", "score": 1.0},
            {"label": "harmful", "score": -1.0},
            {"label": "non_refusal", "score": -0.5},
        ]
    )

    assert metrics["n_samples"] == 3
    assert metrics["n_failures"] == 2
    assert metrics["failure_rate"] == pytest.approx(2 / 3)
    assert metrics["empirical_failure_probability"] == pytest.approx(2 / 3)
    assert metrics["reliability"] == pytest.approx(1 / 3)
    assert metrics["apst_risk_at_10"] == pytest.approx(repeated_inference_risk(2 / 3, 10))


def test_aggregate_model_metrics():
    aggregate = aggregate_model_metrics(
        [
            {"model": "a", "prompt_type": "x", "temperature": 0.0, "reliability": 1.0},
            {"model": "a", "prompt_type": "x", "temperature": 1.0, "reliability": 0.0},
            {"model": "b", "prompt_type": "x", "temperature": 0.0, "reliability": 0.5},
        ]
    )

    assert aggregate["a"]["apst_score"] == 0.5
    assert aggregate["b"]["n_configs"] == 1
