"""APST-style repeated-sampling experiment runner."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from llm_eval.config import DEFAULT_JUDGE_MODEL, EvaluationConfig, load_env
from llm_eval.experiments.checkpoint import CheckpointStore, config_key, latest_checkpoint
from llm_eval.generation.sampler import sample_responses
from llm_eval.metrics import aggregate_model_metrics, calculate_reliability
from llm_eval.models.factory import create_model_runner
from llm_eval.prompts import PromptRegistry, PromptSpec
from llm_eval.results import write_apst_results_csv
from llm_eval.scoring.classifier import (
    classify_response_with_mode,
    generation_error_classification,
)

logger = logging.getLogger(__name__)


def run_apst(config: EvaluationConfig, *, resume: bool = True) -> dict[str, Any]:
    """Run repeated sampling and judge each response."""

    load_env()
    prompts = _load_prompts(config)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    checkpoint_path = _checkpoint_path(config, output_dir, timestamp, resume=resume)
    if config.result_name:
        result_stem = config.result_name
    elif checkpoint_path.name.startswith("apst_checkpoint_"):
        timestamp = checkpoint_path.stem.replace("apst_checkpoint_", "")
        result_stem = f"apst_results_{timestamp}"
    else:
        result_stem = f"apst_results_{timestamp}"

    results_path = output_dir / f"{result_stem}.json"
    csv_path = output_dir / f"{result_stem}.csv"
    store = (
        CheckpointStore.load(checkpoint_path)
        if resume
        else CheckpointStore(path=checkpoint_path)
    )

    model_runners = {
        model_config.name: create_model_runner(
            model_config,
            max_retries=config.max_retries,
            retry_delay=config.retry_delay,
            max_concurrent_requests=config.max_concurrent_requests,
            max_requests_per_second=config.max_requests_per_second,
        )
        for model_config in config.models
    }

    judge_model = None
    if config.judge_mode == "llm":
        judge_config = config.judge_model or DEFAULT_JUDGE_MODEL
        judge_model = create_model_runner(
            judge_config,
            max_retries=config.max_retries,
            retry_delay=config.retry_delay,
            max_concurrent_requests=config.max_concurrent_requests,
            max_requests_per_second=config.max_requests_per_second,
        )

    run_config = _config_summary(config, prompts)
    total_configs = len(model_runners) * len(prompts) * len(config.temperatures)
    processed = 0
    logger.info("Starting APST run: %s configs", total_configs)

    for model_name, runner in model_runners.items():
        model_config = next(item for item in config.models if item.name == model_name)
        for prompt in prompts:
            for temperature in config.temperatures:
                processed += 1
                key = config_key(model_name, prompt.id, temperature, config.n_samples_per_prompt)
                if key in store.completed_keys:
                    logger.info("[%s/%s] skip %s", processed, total_configs, key)
                    continue

                max_tokens = config.max_tokens_per_response
                if model_config.max_tokens is not None:
                    max_tokens = model_config.max_tokens

                outputs = sample_responses(
                    runner,
                    prompt.text,
                    temperature=temperature,
                    n_samples=config.n_samples_per_prompt,
                    max_tokens=max_tokens,
                    max_workers=config.max_workers,
                    batch_size=config.batch_size,
                    delay_between_batches=config.delay_between_batches,
                    show_progress=config.n_samples_per_prompt > 1,
                    **model_config.extra_params,
                )

                classifications = []
                raw_entries = []
                for sample_index, output in enumerate(outputs):
                    if output.error:
                        classification = generation_error_classification(output.error)
                    else:
                        classification = classify_response_with_mode(
                            prompt=prompt.text,
                            response=output.text,
                            judge_mode=config.judge_mode,
                            judge_model=judge_model,
                            use_heuristics=config.use_heuristics,
                            simple_prompt=config.simple_judge_prompt,
                        )
                    classifications.append(classification)

                    if config.save_raw_responses or config.save_classifications:
                        raw_entries.append(
                            _raw_entry(
                                key=key,
                                model=model_name,
                                prompt=prompt,
                                temperature=temperature,
                                sample_index=sample_index,
                                output=output,
                                classification=classification,
                                save_response=config.save_raw_responses,
                                save_classification=config.save_classifications,
                            )
                        )

                metrics = calculate_reliability(
                    classifications,
                    risk_horizon=config.apst_risk_horizon,
                )
                result = {
                    "model": model_name,
                    "prompt_id": prompt.id,
                    "prompt_type": prompt.prompt_type.value,
                    "prompt_domain": prompt.domain,
                    "temperature": temperature,
                    "n_samples": config.n_samples_per_prompt,
                    **metrics,
                }
                store.mark_complete(result=result, raw_entries=raw_entries)
                store.save(config=run_config, partial=True)
                logger.info(
                    "[%s/%s] done %s reliability=%.3f",
                    processed,
                    total_configs,
                    key,
                    result["reliability"],
                )

    model_metrics = aggregate_model_metrics(store.results)
    final_payload = {
        "partial": False,
        "finalized": True,
        "timestamp": timestamp,
        "finalized_timestamp": datetime.now().isoformat(),
        "experiment_type": "apst_repeated_sampling",
        "experiment_config": run_config,
        "results": store.results,
        "model_metrics": model_metrics,
        "artifacts": {
            "results_json": str(results_path),
            "results_csv": str(csv_path),
            "checkpoint": str(checkpoint_path),
        },
    }
    if config.save_raw_responses or config.save_classifications:
        final_payload["raw_responses"] = store.raw_responses

    results_path.write_text(json.dumps(final_payload, indent=2), encoding="utf-8")
    write_apst_results_csv(final_payload, csv_path)
    store.save(config=run_config, partial=False)
    logger.info("APST results saved to %s", results_path)
    return final_payload


def _checkpoint_path(
    config: EvaluationConfig,
    output_dir: Path,
    timestamp: str,
    *,
    resume: bool,
) -> Path:
    if config.result_name:
        return output_dir / f"{config.result_name}_checkpoint.json"
    if resume:
        checkpoint = latest_checkpoint(output_dir, "apst")
        if checkpoint is not None:
            return checkpoint
    return output_dir / f"apst_checkpoint_{timestamp}.json"


def _load_prompts(config: EvaluationConfig) -> list[PromptSpec]:
    if not config.prompt_set_path:
        raise ValueError(
            "prompt_set_path is required. Freeze AIRBench or provide a custom JSON file."
        )

    registry = PromptRegistry()
    registry.load_prompt_set(config.prompt_set_path)
    prompts = registry.all()
    if config.max_prompts is not None:
        prompts = prompts[: config.max_prompts]
    if not prompts:
        raise ValueError("No prompts loaded")
    logger.info("Loaded %s prompts from %s", len(prompts), config.prompt_set_path)
    return prompts


def _raw_entry(
    *,
    key: str,
    model: str,
    prompt: PromptSpec,
    temperature: float,
    sample_index: int,
    output,
    classification: dict[str, Any],
    save_response: bool,
    save_classification: bool,
) -> dict[str, Any]:
    return {
        "config_key": key,
        "model": model,
        "prompt_id": prompt.id,
        "prompt_type": prompt.prompt_type.value,
        "temperature": temperature,
        "sample_index": sample_index,
        "response": output.text if save_response else None,
        "classification": classification if save_classification else None,
        "generation_error": output.error,
        "finish_reason": output.finish_reason,
        "input_tokens": output.input_tokens,
        "output_tokens": output.output_tokens,
    }


def _config_summary(config: EvaluationConfig, prompts: list[PromptSpec]) -> dict[str, Any]:
    return {
        "models": [model.name for model in config.models],
        "judge_mode": config.judge_mode,
        "judge_model": (config.judge_model or DEFAULT_JUDGE_MODEL).name
        if config.judge_mode == "llm"
        else None,
        "prompt_set_path": config.prompt_set_path,
        "n_prompts": len(prompts),
        "temperatures": config.temperatures,
        "n_samples_per_prompt": config.n_samples_per_prompt,
        "apst_risk_horizon": config.apst_risk_horizon,
        "max_workers": config.max_workers,
    }
