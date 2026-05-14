"""Configuration loading for evaluation runs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - dependency is declared, guard is for friendly errors.
    yaml = None

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for a model under test or judge model."""

    name: str
    model_id: str
    provider: str
    api_key: str | None = None
    base_url: str | None = None
    max_tokens: int | None = None
    extra_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationConfig:
    """Configuration for APST and AIRBench-style evaluations."""

    models: list[ModelConfig] = field(default_factory=list)
    judge_model: ModelConfig | None = None
    judge_mode: str = "llm"
    prompt_set_path: str | None = None
    output_dir: str = "data/results"
    result_name: str | None = None

    temperatures: list[float] = field(default_factory=lambda: [0.0, 0.7, 1.0])
    n_samples_per_prompt: int = 10
    max_prompts: int | None = None
    max_tokens_per_response: int | None = 512
    apst_risk_horizon: int = 10

    max_workers: int = 1
    batch_size: int = 10
    delay_between_batches: float = 0.0
    max_retries: int = 3
    retry_delay: float = 1.0

    max_concurrent_requests: int = 5
    max_requests_per_second: float = 2.0

    save_raw_responses: bool = True
    save_classifications: bool = True
    use_heuristics: bool = True
    simple_judge_prompt: bool = False


DEFAULT_JUDGE_MODEL = ModelConfig(
    name="gpt-4o-mini-judge",
    model_id="gpt-4o-mini",
    provider="openai",
    max_tokens=256,
)


def load_env(env_path: str | Path | None = None) -> None:
    """Load `.env` without overriding already-exported environment variables."""

    if load_dotenv is None:
        return
    load_dotenv(dotenv_path=env_path, override=False)


def load_config(path: str | Path) -> EvaluationConfig:
    """Load a YAML or JSON evaluation config."""

    config_path = Path(path)
    data = _load_mapping(config_path)
    return config_from_dict(data)


def config_from_dict(data: dict[str, Any]) -> EvaluationConfig:
    """Convert a mapping into an :class:`EvaluationConfig`."""

    models = [_model_config_from_dict(item) for item in data.get("models", [])]
    judge = data.get("judge_model")

    return EvaluationConfig(
        models=models,
        judge_model=_model_config_from_dict(judge) if judge else None,
        judge_mode=str(data.get("judge_mode", "llm")).lower(),
        prompt_set_path=data.get("prompt_set_path"),
        output_dir=data.get("output_dir", "data/results"),
        result_name=data.get("result_name"),
        temperatures=[float(t) for t in data.get("temperatures", [0.0, 0.7, 1.0])],
        n_samples_per_prompt=int(data.get("n_samples_per_prompt", 10)),
        max_prompts=_optional_int(data.get("max_prompts")),
        max_tokens_per_response=_optional_int(data.get("max_tokens_per_response", 512)),
        apst_risk_horizon=int(data.get("apst_risk_horizon", 10)),
        max_workers=int(data.get("max_workers", 1)),
        batch_size=int(data.get("batch_size", 10)),
        delay_between_batches=float(data.get("delay_between_batches", 0.0)),
        max_retries=int(data.get("max_retries", 3)),
        retry_delay=float(data.get("retry_delay", 1.0)),
        max_concurrent_requests=int(data.get("max_concurrent_requests", 5)),
        max_requests_per_second=float(data.get("max_requests_per_second", 2.0)),
        save_raw_responses=bool(data.get("save_raw_responses", True)),
        save_classifications=bool(data.get("save_classifications", True)),
        use_heuristics=bool(data.get("use_heuristics", True)),
        simple_judge_prompt=bool(data.get("simple_judge_prompt", False)),
    )


def apply_cli_overrides(
    config: EvaluationConfig,
    *,
    models: list[str] | None = None,
    prompt_set_path: str | None = None,
    temperatures: list[float] | None = None,
    n_samples: int | None = None,
    max_prompts: int | None = None,
    output_dir: str | None = None,
    judge_mode: str | None = None,
    result_name: str | None = None,
) -> EvaluationConfig:
    """Apply common command-line overrides to a loaded config."""

    if models:
        from llm_eval.models.factory import infer_provider

        config.models = [
            ModelConfig(name=model, model_id=model, provider=infer_provider(model))
            for model in models
        ]
    if prompt_set_path:
        config.prompt_set_path = prompt_set_path
    if temperatures:
        config.temperatures = temperatures
    if n_samples is not None:
        config.n_samples_per_prompt = n_samples
    if max_prompts is not None:
        config.max_prompts = max_prompts
    if output_dir:
        config.output_dir = output_dir
    if judge_mode:
        config.judge_mode = judge_mode.lower()
    if result_name:
        config.result_name = result_name
    return config


def _load_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required to load YAML configs")
        loaded = yaml.safe_load(path.read_text()) or {}
    elif path.suffix.lower() == ".json":
        loaded = json.loads(path.read_text())
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")

    if not isinstance(loaded, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return loaded


def _model_config_from_dict(data: dict[str, Any]) -> ModelConfig:
    if not isinstance(data, dict):
        raise TypeError("Model config entries must be mappings")
    return ModelConfig(
        name=str(data["name"]),
        model_id=str(data.get("model_id", data["name"])),
        provider=str(data["provider"]).lower(),
        api_key=data.get("api_key"),
        base_url=data.get("base_url"),
        max_tokens=_optional_int(data.get("max_tokens")),
        extra_params=dict(data.get("extra_params", {})),
    )


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
