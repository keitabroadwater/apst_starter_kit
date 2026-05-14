"""Compare APST and AIRBench result files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def compare_results(apst_path: str | Path, airbench_path: str | Path) -> dict[str, Any]:
    apst = json.loads(Path(apst_path).read_text())
    airbench = json.loads(Path(airbench_path).read_text())

    apst_scores = {
        model: float(metrics.get("apst_score", 0.0))
        for model, metrics in apst.get("model_metrics", {}).items()
    }
    airbench_scores = {
        model: float(metrics.get("refusal_rate", metrics.get("mean_score", 0.0)))
        for model, metrics in airbench.get("model_metrics", {}).items()
    }
    common_models = sorted(set(apst_scores) & set(airbench_scores))

    correlation = None
    if len(common_models) >= 2:
        x = [apst_scores[model] for model in common_models]
        y = [airbench_scores[model] for model in common_models]
        correlation = float(np.corrcoef(x, y)[0, 1])

    return {
        "common_models": common_models,
        "correlation_apst_vs_airbench_refusal": correlation,
        "scores": {
            model: {
                "apst_score": apst_scores[model],
                "airbench_refusal_rate": airbench_scores[model],
            }
            for model in common_models
        },
        "rankings": {
            "apst": _ranking(apst_scores, common_models),
            "airbench_refusal": _ranking(airbench_scores, common_models),
        },
    }


def _ranking(scores: dict[str, float], models: list[str]) -> dict[str, int]:
    ranked = sorted(models, key=lambda model: scores[model], reverse=True)
    return {model: index + 1 for index, model in enumerate(ranked)}

