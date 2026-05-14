"""Repeated sampling helpers."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from tqdm import tqdm

from llm_eval.models.base import GenerationOutput, ModelRunner

logger = logging.getLogger(__name__)


def sample_responses(
    model: ModelRunner,
    prompt: str,
    *,
    temperature: float,
    n_samples: int,
    max_tokens: int | None = None,
    max_workers: int = 1,
    batch_size: int = 10,
    delay_between_batches: float = 0.0,
    show_progress: bool = True,
    **generation_kwargs: Any,
) -> list[GenerationOutput]:
    """Generate N responses for the same prompt and preserve output order."""

    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")

    if max_workers <= 1:
        return _sample_sequential(
            model,
            prompt,
            temperature=temperature,
            n_samples=n_samples,
            max_tokens=max_tokens,
            batch_size=batch_size,
            delay_between_batches=delay_between_batches,
            show_progress=show_progress,
            **generation_kwargs,
        )

    outputs: list[GenerationOutput | None] = [None] * n_samples
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                model.generate,
                prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                **generation_kwargs,
            ): index
            for index in range(n_samples)
        }
        progress = tqdm(total=n_samples, desc=f"sampling {model.model_name}") if show_progress else None
        try:
            for future in as_completed(futures):
                index = futures[future]
                try:
                    outputs[index] = future.result()
                except Exception as exc:  # noqa: BLE001 - defensive around runner bugs.
                    logger.exception("Generation worker crashed")
                    outputs[index] = GenerationOutput(
                        text="",
                        model_name=model.model_name,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                if progress:
                    progress.update(1)
        finally:
            if progress:
                progress.close()

    return [output if output is not None else _missing_output(model) for output in outputs]


def _sample_sequential(
    model: ModelRunner,
    prompt: str,
    *,
    temperature: float,
    n_samples: int,
    max_tokens: int | None,
    batch_size: int,
    delay_between_batches: float,
    show_progress: bool,
    **generation_kwargs: Any,
) -> list[GenerationOutput]:
    outputs: list[GenerationOutput] = []
    iterator = range(n_samples)
    progress = tqdm(iterator, desc=f"sampling {model.model_name}") if show_progress else iterator
    for index in progress:
        if delay_between_batches > 0 and index > 0 and index % batch_size == 0:
            time.sleep(delay_between_batches)
        outputs.append(
            model.generate(
                prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                **generation_kwargs,
            )
        )
    return outputs


def _missing_output(model: ModelRunner) -> GenerationOutput:
    return GenerationOutput(text="", model_name=model.model_name, error="generation future missing")

