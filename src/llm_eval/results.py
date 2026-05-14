"""Result-file helpers for APST runs and reports."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


PREFERRED_RESULT_COLUMNS = [
    "model",
    "prompt_id",
    "prompt_type",
    "prompt_domain",
    "temperature",
    "n_samples",
    "n_failures",
    "failure_rate",
    "empirical_failure_probability",
    "reliability",
    "apst_risk_horizon",
    "apst_risk_at_10",
    "failure_probability_ci_low",
    "failure_probability_ci_high",
    "mean_score",
    "std_score",
    "failure_mode_distribution",
]


def write_apst_results_csv(payload: dict[str, Any], output_path: str | Path) -> Path:
    """Write final APST result rows to CSV."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [dict(row) for row in payload.get("results", [])]
    fieldnames = _fieldnames(rows)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fieldnames})
    return path


def load_result_rows(path: str | Path) -> list[dict[str, Any]]:
    """Load APST rows from a final JSON payload or CSV result file."""

    result_path = Path(path)
    if result_path.suffix.lower() == ".json":
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        return [dict(row) for row in payload.get("results", [])]
    if result_path.suffix.lower() == ".csv":
        with result_path.open(newline="", encoding="utf-8") as handle:
            return [_coerce_row(row) for row in csv.DictReader(handle)]
    raise ValueError(f"Unsupported results format: {result_path.suffix}")


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return PREFERRED_RESULT_COLUMNS
    keys = {key for row in rows for key in row}
    preferred = [key for key in PREFERRED_RESULT_COLUMNS if key in keys or key == "apst_risk_at_10"]
    extras = sorted(keys - set(preferred))
    return preferred + extras


def _csv_value(value: Any) -> Any:
    if isinstance(value, dict | list):
        return json.dumps(value, sort_keys=True)
    if value is None:
        return ""
    return value


def _coerce_row(row: dict[str, str]) -> dict[str, Any]:
    return {key: _coerce_value(value) for key, value in row.items()}


def _coerce_value(value: str) -> Any:
    if value == "":
        return None
    stripped = value.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return value
    try:
        number = float(stripped)
    except ValueError:
        return value
    if number.is_integer() and "." not in stripped:
        return int(number)
    return number
