#!/usr/bin/env python3
"""Freeze a stratified AIRBench prompt set."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_eval.datasets.airbench import freeze_airbench_prompt_set


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze AIRBench prompts to JSON")
    parser.add_argument("--region", default="us")
    parser.add_argument("--split", default="test")
    parser.add_argument("--per-l4", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/prompts/airbench_us_v1.json")
    args = parser.parse_args()

    output = freeze_airbench_prompt_set(
        region=args.region,
        split=args.split,
        per_l4=args.per_l4,
        seed=args.seed,
        output_path=args.output,
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

