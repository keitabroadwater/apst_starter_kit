"""Small JSON checkpoint store."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


def config_key(model: str, prompt_id: str, temperature: float, n_samples: int | None = None) -> str:
    if n_samples is None:
        return f"{model}::{prompt_id}::{temperature}"
    return f"{model}::{prompt_id}::{temperature}::{n_samples}"


@dataclass
class CheckpointStore:
    """Append-only checkpoint data for resumable experiments."""

    path: Path
    results: list[dict[str, Any]] = field(default_factory=list)
    raw_responses: list[dict[str, Any]] = field(default_factory=list)
    completed_keys: set[str] = field(default_factory=set)

    @classmethod
    def load(cls, path: str | Path) -> "CheckpointStore":
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            return cls(path=checkpoint_path)

        data = json.loads(checkpoint_path.read_text())
        results = list(data.get("results", []))
        raw_responses = list(data.get("raw_responses", []))
        completed = {
            config_key(
                str(result["model"]),
                str(result["prompt_id"]),
                float(result["temperature"]),
                int(result["n_samples"]) if result.get("n_samples") is not None else None,
            )
            for result in results
        }
        return cls(checkpoint_path, results, raw_responses, completed)

    def mark_complete(
        self,
        *,
        result: dict[str, Any],
        raw_entries: list[dict[str, Any]],
    ) -> None:
        key = config_key(
            str(result["model"]),
            str(result["prompt_id"]),
            float(result["temperature"]),
            int(result["n_samples"]) if result.get("n_samples") is not None else None,
        )
        if key in self.completed_keys:
            return
        self.results.append(result)
        self.raw_responses.extend(raw_entries)
        self.completed_keys.add(key)

    def save(self, *, config: dict[str, Any] | None = None, partial: bool = True) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "partial": partial,
            "checkpoint_timestamp": datetime.now().isoformat(),
            "experiment_config": config or {},
            "n_results": len(self.results),
            "results": self.results,
            "raw_responses": self.raw_responses,
        }
        temp_path = self.path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(payload, indent=2))
        temp_path.replace(self.path)


def latest_checkpoint(output_dir: str | Path, prefix: str) -> Path | None:
    output_path = Path(output_dir)
    candidates = list(output_path.glob(f"{prefix}_checkpoint_*.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)

