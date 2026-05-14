"""AIRBench prompt and judge-prompt helpers."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from llm_eval.prompts import PromptSpec, PromptType, SourceBenchmark


AIRBENCH_L2_TO_PROMPT_TYPE: dict[str, PromptType] = {
    "Security Risks": PromptType.SECURITY_PRIVACY,
    "Operational Misuses": PromptType.SAFETY_HARMFUL,
    "Content Safety": PromptType.SAFETY_HARMFUL,
    "Societal Risks": PromptType.SAFETY_CRITICAL,
    "Legal Risks": PromptType.LEGAL_INFO,
}

AIRBENCH_L3_TO_DOMAIN: dict[str, str] = {
    "Confidentiality": "security",
    "Integrity": "security",
    "Availability": "security",
    "Fraud": "finance",
    "Financial Crime": "finance",
    "Influence Operations": "politics",
    "Defamation": "legal",
    "Privacy": "privacy",
    "Hate Speech": "content_moderation",
    "Sexual Content": "content_moderation",
    "Violence": "content_moderation",
    "Self-harm": "healthcare",
    "Discrimination": "social",
    "Child Safety": "safety",
    "Harassment": "social",
    "Weapons": "security",
    "Drugs": "healthcare",
}


def map_l2_to_prompt_type(l2_name: str) -> PromptType:
    return AIRBENCH_L2_TO_PROMPT_TYPE.get(l2_name, PromptType.SAFETY_HARMFUL)


def map_l3_to_domain(l3_name: str) -> str:
    return AIRBENCH_L3_TO_DOMAIN.get(l3_name, "generic")


def estimate_risk_level(l2_name: str, l3_name: str) -> str:
    high_risk_l3 = {"Child Safety", "Weapons", "Violence", "Self-harm", "Fraud"}
    medium_risk_l3 = {"Confidentiality", "Harassment", "Hate Speech", "Privacy"}

    if l3_name in high_risk_l3:
        return "5"
    if l3_name in medium_risk_l3 or l2_name == "Security Risks":
        return "4"
    return "3"


def load_airbench_prompts(
    *,
    subset: str = "us",
    split: str = "test",
    max_prompts: int | None = None,
) -> list[PromptSpec]:
    """Load AIRBench prompts directly from Hugging Face."""

    load_dataset = _load_dataset()
    dataset = load_dataset("stanford-crfm/air-bench-2024", subset, split=split)

    prompts: list[PromptSpec] = []
    for index, row in enumerate(dataset):
        if max_prompts is not None and len(prompts) >= max_prompts:
            break

        text = str(row.get("prompt", ""))
        if not text.strip():
            continue

        l2 = str(row.get("l2-name", ""))
        l3 = str(row.get("l3-name", ""))
        l4 = str(row.get("l4-name", ""))
        cate_idx = str(row.get("cate-idx", ""))

        prompts.append(
            PromptSpec(
                id=f"{cate_idx}::{index}",
                text=text,
                prompt_type=map_l2_to_prompt_type(l2),
                source=SourceBenchmark.AIRBENCH,
                domain=map_l3_to_domain(l3),
                risk_level=estimate_risk_level(l2, l3),
                notes=f"L2: {l2} | L3: {l3} | L4: {l4}",
                metadata={
                    "cate_idx": cate_idx,
                    "l2": l2,
                    "l3": l3,
                    "l4": l4,
                    "region": subset,
                },
            )
        )
    return prompts


def load_airbench_judge_prompts() -> dict[str, str]:
    """Load AIRBench category-specific judge templates."""

    load_dataset = _load_dataset()
    dataset = load_dataset("stanford-crfm/air-bench-2024", "judge_prompts", split="test")
    return {
        str(row.get("cate-idx", "")): str(row.get("judge_prompt", ""))
        for row in dataset
        if row.get("cate-idx") and row.get("judge_prompt")
    }


def freeze_airbench_prompt_set(
    *,
    region: str = "us",
    split: str = "test",
    per_l4: int = 10,
    seed: int = 42,
    output_path: str | Path = "data/prompts/airbench_us_v1.json",
) -> Path:
    """Export a stratified AIRBench prompt set for reproducible experiments."""

    load_dataset = _load_dataset()
    dataset = load_dataset("stanford-crfm/air-bench-2024", region, split=split)
    rows = [dict(row) for row in dataset]

    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(str(row.get("l4-name", "unknown")), []).append(row)

    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for items in buckets.values():
        items = list(items)
        rng.shuffle(items)
        selected.extend(items[:per_l4])

    payload = {
        "source": "stanford-crfm/air-bench-2024",
        "region": region,
        "split": split,
        "seed": seed,
        "per_l4": per_l4,
        "count": len(selected),
        "prompts": [
            {
                "id": f'{row["cate-idx"]}::{index}',
                "cate_idx": row["cate-idx"],
                "l2": row["l2-name"],
                "l3": row["l3-name"],
                "l4": row["l4-name"],
                "prompt": row["prompt"],
            }
            for index, row in enumerate(selected)
        ],
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    return output


def _load_dataset():
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Install the airbench extra to use AIRBench helpers: pip install -e '.[airbench]'"
        ) from exc
    return load_dataset

