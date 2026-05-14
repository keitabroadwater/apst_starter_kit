"""Prompt models and prompt-set loading."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable


class PromptType(str, Enum):
    """Broad prompt categories used for aggregation."""

    SAFETY_HARMFUL = "safety_harmful"
    SAFETY_CRITICAL = "safety_critical"
    MEDICAL_INFO = "medical_info"
    FINANCIAL_INFO = "financial_info"
    LEGAL_INFO = "legal_info"
    SECURITY_PRIVACY = "security_privacy"
    GENERAL_KNOWLEDGE = "general_knowledge"
    PROMPT_INJECTION = "prompt_injection"
    CUSTOM = "custom"


class SourceBenchmark(str, Enum):
    """Known prompt sources."""

    AIRBENCH = "airbench2024"
    CUSTOM = "custom"
    OTHER = "other"


@dataclass(frozen=True)
class PromptSpec:
    """A single prompt plus metadata needed for grouping and judge lookup."""

    id: str
    text: str
    prompt_type: PromptType
    source: SourceBenchmark
    domain: str = "generic"
    risk_level: str = "3"
    notes: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "prompt_type": self.prompt_type.value,
            "source": self.source.value,
            "domain": self.domain,
            "risk_level": self.risk_level,
            "notes": self.notes,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PromptSpec":
        return cls(
            id=str(data["id"]),
            text=str(data.get("text", data.get("prompt", ""))),
            prompt_type=_prompt_type(data.get("prompt_type", "custom")),
            source=_source(data.get("source", "custom")),
            domain=str(data.get("domain", "generic")),
            risk_level=str(data.get("risk_level", "3")),
            notes=data.get("notes"),
            metadata=dict(data.get("metadata", {})),
        )


class PromptRegistry:
    """Small registry for loading and filtering prompts."""

    def __init__(self) -> None:
        self._prompts: list[PromptSpec] = []
        self._by_id: dict[str, PromptSpec] = {}

    def add(self, prompt: PromptSpec) -> None:
        if prompt.id in self._by_id:
            raise ValueError(f"Duplicate prompt id: {prompt.id}")
        self._prompts.append(prompt)
        self._by_id[prompt.id] = prompt

    def add_many(self, prompts: Iterable[PromptSpec]) -> None:
        for prompt in prompts:
            self.add(prompt)

    def load_prompt_set(self, path: str | Path) -> int:
        """Load a frozen JSON prompt set."""

        prompt_set_path = Path(path)
        payload = json.loads(prompt_set_path.read_text())
        prompts = prompts_from_payload(payload)
        self.add_many(prompts)
        return len(prompts)

    def all(self) -> list[PromptSpec]:
        return list(self._prompts)

    def get(self, prompt_id: str) -> PromptSpec | None:
        return self._by_id.get(prompt_id)

    def filter_by_type(self, prompt_type: PromptType) -> list[PromptSpec]:
        return [prompt for prompt in self._prompts if prompt.prompt_type == prompt_type]

    def stats(self) -> dict[str, Any]:
        return {
            "total": len(self._prompts),
            "by_type": dict(Counter(prompt.prompt_type.value for prompt in self._prompts)),
            "by_domain": dict(Counter(prompt.domain for prompt in self._prompts)),
            "by_source": dict(Counter(prompt.source.value for prompt in self._prompts)),
            "by_risk_level": dict(Counter(prompt.risk_level for prompt in self._prompts)),
        }

    def __len__(self) -> int:
        return len(self._prompts)

    def __iter__(self):
        return iter(self._prompts)


def prompts_from_payload(payload: dict[str, Any] | list[dict[str, Any]]) -> list[PromptSpec]:
    """Convert supported JSON prompt-set shapes into PromptSpec objects."""

    if isinstance(payload, list):
        raw_prompts = payload
        top_level_source = "custom"
        region = None
    else:
        raw_prompts = payload.get("prompts", [])
        top_level_source = payload.get("source", "custom")
        region = payload.get("region")

    prompts: list[PromptSpec] = []
    for index, row in enumerate(raw_prompts):
        if "text" in row and "prompt_type" in row:
            prompt = PromptSpec.from_dict(row)
        else:
            prompt = _prompt_from_benchmark_row(row, index, top_level_source, region)
        prompts.append(prompt)
    return prompts


def _prompt_from_benchmark_row(
    row: dict[str, Any],
    index: int,
    top_level_source: str,
    region: str | None,
) -> PromptSpec:
    from llm_eval.datasets.airbench import (
        estimate_risk_level,
        map_l2_to_prompt_type,
        map_l3_to_domain,
    )

    l2 = str(row.get("l2", row.get("l2-name", "")))
    l3 = str(row.get("l3", row.get("l3-name", "")))
    l4 = str(row.get("l4", row.get("l4-name", "")))
    cate_idx = str(row.get("cate_idx", row.get("cate-idx", "")))
    text = str(row.get("prompt", row.get("text", "")))

    if not text.strip():
        raise ValueError(f"Prompt row {index} has empty text")

    source = (
        SourceBenchmark.AIRBENCH
        if "air-bench" in top_level_source or "airbench" in top_level_source.lower()
        else SourceBenchmark.CUSTOM
    )

    return PromptSpec(
        id=str(row.get("id", f"{cate_idx or 'prompt'}::{index}")),
        text=text,
        prompt_type=map_l2_to_prompt_type(l2) if l2 else PromptType.CUSTOM,
        source=source,
        domain=map_l3_to_domain(l3) if l3 else "generic",
        risk_level=estimate_risk_level(l2, l3) if (l2 or l3) else "3",
        notes=f"L2: {l2} | L3: {l3} | L4: {l4}",
        metadata={
            "cate_idx": cate_idx,
            "l2": l2,
            "l3": l3,
            "l4": l4,
            "region": region,
            **dict(row.get("metadata", {})),
        },
    )


def _prompt_type(value: str) -> PromptType:
    try:
        return PromptType(str(value))
    except ValueError:
        return PromptType.CUSTOM


def _source(value: str) -> SourceBenchmark:
    try:
        return SourceBenchmark(str(value))
    except ValueError:
        return SourceBenchmark.OTHER

