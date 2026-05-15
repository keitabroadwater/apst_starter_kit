"""Command-line interface."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from llm_eval.config import apply_cli_overrides, load_config
from llm_eval.datasets.airbench import freeze_airbench_prompt_set
from llm_eval.experiments.airbench import run_airbench
from llm_eval.experiments.apst import run_apst
from llm_eval.experiments.compare import compare_results
from llm_eval.models.pricing import estimate_experiment_cost
from llm_eval.report import generate_report
from llm_eval.starter import init_starter_workspace


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM reliability evaluation tools")
    parser.add_argument("--log-level", default="INFO")

    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_run(subparsers)
    _add_run_apst(subparsers)
    _add_run_airbench(subparsers)
    _add_report(subparsers)
    _add_freeze_airbench(subparsers)
    _add_compare(subparsers)
    _add_estimate_cost(subparsers)
    _add_init(subparsers)

    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    if args.command in {"run", "run-apst"}:
        config = _config_with_overrides(args)
        payload = run_apst(config, resume=not args.no_resume)
        print(json.dumps(payload.get("artifacts", {}), indent=2))
    elif args.command == "run-airbench":
        config = _config_with_overrides(args)
        run_airbench(config, temperature=args.temperature)
    elif args.command == "report":
        report_path = generate_report(
            results_path=args.results,
            lang=args.lang,
            output_path=args.output,
            title=args.title,
            audit_contact=args.audit_contact,
        )
        print(report_path)
    elif args.command == "freeze-airbench":
        output = freeze_airbench_prompt_set(
            region=args.region,
            split=args.split,
            per_l4=args.per_l4,
            seed=args.seed,
            output_path=args.output,
        )
        print(output)
    elif args.command == "compare":
        comparison = compare_results(args.apst_results, args.airbench_results)
        if args.output:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(comparison, indent=2))
        print(json.dumps(comparison, indent=2))
    elif args.command == "estimate-cost":
        estimate = estimate_experiment_cost(
            models=args.models,
            n_prompts=args.n_prompts,
            n_temperatures=args.n_temperatures,
            n_samples=args.n_samples,
            judge_model=args.judge_model,
        )
        print(json.dumps(estimate, indent=2))
    elif args.command == "init":
        target = init_starter_workspace(args.directory, force=args.force)
        print(f"Initialized APST starter workspace at {target}")
        print("Next: cd into that directory and run `apst run --config configs/demo_mock.yaml`")


def _add_run(subparsers) -> None:
    parser = subparsers.add_parser("run", help="Run APST repeated-sampling evaluation")
    _add_config_args(parser)
    parser.add_argument("--no-resume", action="store_true", help="Start a fresh run")


def _add_run_apst(subparsers) -> None:
    parser = subparsers.add_parser("run-apst", help="Run APST repeated-sampling evaluation")
    _add_config_args(parser)
    parser.add_argument("--no-resume", action="store_true", help="Start a fresh run")


def _add_run_airbench(subparsers) -> None:
    parser = subparsers.add_parser("run-airbench", help="Run AIRBench-style one-shot evaluation")
    _add_config_args(parser)
    parser.add_argument("--temperature", type=float, default=0.0)


def _add_config_args(parser) -> None:
    parser.add_argument("--config", required=True, help="YAML or JSON config path")
    parser.add_argument("--models", nargs="+", help="Override model IDs")
    parser.add_argument("--prompt-set", dest="prompt_set_path", help="Override prompt set path")
    parser.add_argument("--temperatures", nargs="+", type=float)
    parser.add_argument("--n-samples", type=int)
    parser.add_argument("--max-prompts", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--judge-mode", choices=["llm", "rule", "heuristic"])
    parser.add_argument("--result-name")


def _add_report(subparsers) -> None:
    parser = subparsers.add_parser("report", help="Generate an APST Markdown report")
    parser.add_argument("--results", required=True, help="APST CSV or JSON results path")
    parser.add_argument("--lang", choices=["en", "zh", "both"], default="en")
    parser.add_argument("--output", help="Report output path")
    parser.add_argument("--title", default="APST Starter Kit Report")
    parser.add_argument(
        "--audit-contact",
        help="Audit intake URL, email, or conference contact note",
    )


def _add_freeze_airbench(subparsers) -> None:
    parser = subparsers.add_parser("freeze-airbench", help="Export frozen AIRBench prompt set")
    parser.add_argument("--region", default="us")
    parser.add_argument("--split", default="test")
    parser.add_argument("--per-l4", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/prompts/airbench_us_v1.json")


def _add_compare(subparsers) -> None:
    parser = subparsers.add_parser("compare", help="Compare APST and AIRBench result files")
    parser.add_argument("--apst-results", required=True)
    parser.add_argument("--airbench-results", required=True)
    parser.add_argument("--output")


def _add_estimate_cost(subparsers) -> None:
    parser = subparsers.add_parser("estimate-cost", help="Estimate generation and judge costs")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--n-prompts", type=int, required=True)
    parser.add_argument("--n-temperatures", type=int, required=True)
    parser.add_argument("--n-samples", type=int, required=True)
    parser.add_argument("--judge-model", default="gpt-4o-mini")


def _add_init(subparsers) -> None:
    parser = subparsers.add_parser(
        "init",
        help="Create a runnable APST starter workspace from the packaged template",
    )
    parser.add_argument("directory", nargs="?", default="apst-starter-workspace")
    parser.add_argument("--force", action="store_true", help="Overwrite existing starter files")


def _config_with_overrides(args):
    config = load_config(args.config)
    return apply_cli_overrides(
        config,
        models=args.models,
        prompt_set_path=args.prompt_set_path,
        temperatures=args.temperatures,
        n_samples=args.n_samples,
        max_prompts=args.max_prompts,
        output_dir=args.output_dir,
        judge_mode=args.judge_mode,
        result_name=args.result_name,
    )


if __name__ == "__main__":
    main()
