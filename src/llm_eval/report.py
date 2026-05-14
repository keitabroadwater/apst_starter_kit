"""Markdown report generation for APST starter-kit runs."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from llm_eval.metrics import repeated_inference_risk
from llm_eval.results import load_result_rows


LANG_CHOICES = {"en", "zh", "both"}


def generate_report(
    *,
    results_path: str | Path,
    lang: str = "en",
    output_path: str | Path | None = None,
    title: str = "APST Starter Kit Report",
    audit_contact: str | None = None,
) -> Path:
    """Generate a Markdown APST report in English, Chinese, or both."""

    normalized_lang = lang.lower()
    if normalized_lang not in LANG_CHOICES:
        raise ValueError(f"Unsupported report language: {lang!r}. Choose one of {LANG_CHOICES}.")

    source_path = Path(results_path)
    rows = load_result_rows(source_path)
    summary = summarize_rows(rows)
    target_path = (
        Path(output_path)
        if output_path
        else _default_report_path(source_path, normalized_lang)
    )
    target_path.parent.mkdir(parents=True, exist_ok=True)

    sections: list[str] = []
    if normalized_lang in {"en", "both"}:
        sections.append(_render_en(title, source_path, summary, audit_contact))
    if normalized_lang in {"zh", "both"}:
        sections.append(_render_zh(title, source_path, summary, audit_contact))

    target_path.write_text("\n\n---\n\n".join(sections), encoding="utf-8")
    return target_path


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Create report-ready aggregate statistics from APST result rows."""

    total_samples = sum(_as_int(row.get("n_samples")) for row in rows)
    total_failures = sum(_as_int(row.get("n_failures")) for row in rows)
    empirical_failure_probability = total_failures / total_samples if total_samples else 0.0
    reliability = 1.0 - empirical_failure_probability
    horizon = _risk_horizon(rows)

    model_rows = _summarize_by_model(rows, horizon=horizon)
    failure_modes = _failure_modes(rows)

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "n_models": len(model_rows),
        "n_configs": len(rows),
        "n_samples": total_samples,
        "n_failures": total_failures,
        "empirical_failure_probability": empirical_failure_probability,
        "apst_score": reliability,
        "risk_horizon": horizon,
        "repeated_inference_risk": repeated_inference_risk(
            empirical_failure_probability,
            horizon,
        ),
        "models": model_rows,
        "failure_modes": failure_modes,
    }


def _summarize_by_model(rows: list[dict[str, Any]], *, horizon: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("model", "unknown"))].append(row)

    summaries = []
    for model, model_results in sorted(grouped.items()):
        n_samples = sum(_as_int(row.get("n_samples")) for row in model_results)
        n_failures = sum(_as_int(row.get("n_failures")) for row in model_results)
        failure_probability = n_failures / n_samples if n_samples else 0.0
        reliabilities = [_as_float(row.get("reliability")) for row in model_results]
        summaries.append(
            {
                "model": model,
                "n_configs": len(model_results),
                "n_samples": n_samples,
                "n_failures": n_failures,
                "empirical_failure_probability": failure_probability,
                "apst_score": float(np.mean(reliabilities)) if reliabilities else 0.0,
                "repeated_inference_risk": repeated_inference_risk(
                    failure_probability,
                    horizon,
                ),
            }
        )
    return summaries


def _failure_modes(rows: list[dict[str, Any]]) -> list[tuple[str, int]]:
    counter: Counter[str] = Counter()
    for row in rows:
        distribution = row.get("failure_mode_distribution") or {}
        if isinstance(distribution, dict):
            for label, count in distribution.items():
                counter[str(label)] += _as_int(count)
    return counter.most_common()


def _render_en(
    title: str,
    source_path: Path,
    summary: dict[str, Any],
    audit_contact: str | None,
) -> str:
    contact = audit_contact or "See docs/enterprise_audits.md or the conference audit intake link."
    lines = [
        f"# {title}",
        "",
        f"Generated: {summary['generated_at']}",
        f"Source results: `{source_path}`",
        "",
        "## Executive Summary",
        "",
        "APST estimates empirical failure probability by repeatedly sampling the same prompts "
        "under controlled conditions. A small nonzero per-sample failure rate can become material "
        "when users, agents, or attackers retry the same task.",
        "",
        _metric_list_en(summary),
        "",
        "## Model Results",
        "",
        _model_table(summary["models"], horizon=summary["risk_horizon"]),
        "",
        "## Failure Modes",
        "",
        _failure_mode_table(summary["failure_modes"]),
        "",
        "## Enterprise Audit Path",
        "",
        "This starter kit is a lightweight local demonstration. A full APST audit usually adds "
        "larger prompt suites, provider-specific controls, calibrated judges, deeper repeated "
        "sampling, and a written risk assessment.",
        "",
        f"Audit requests: {contact}",
    ]
    return "\n".join(lines)


def _render_zh(
    title: str,
    source_path: Path,
    summary: dict[str, Any],
    audit_contact: str | None,
) -> str:
    contact = audit_contact or "请查看 docs/enterprise_audits.md 或会议材料中的审计申请入口。"
    lines = [
        f"# {title}（中文摘要）",
        "",
        f"生成时间：{summary['generated_at']}",
        f"结果来源：`{source_path}`",
        "",
        "## 执行摘要",
        "",
        "APST 通过在受控条件下反复采样同一组提示词，估计模型在重复推理中的经验失败概率。"
        "即使单次失败率很低，当用户、智能体或攻击者反复尝试时，风险也可能被放大。",
        "",
        _metric_list_zh(summary),
        "",
        "## 模型结果",
        "",
        _model_table(summary["models"], horizon=summary["risk_horizon"]),
        "",
        "## 失败模式",
        "",
        _failure_mode_table(summary["failure_modes"]),
        "",
        "## 企业审计路径",
        "",
        "这个 starter kit 是轻量级本地演示。完整 APST 审计通常会加入更大的提示词集、"
        "供应商特定控制、校准后的评审器、更深的重复采样，以及书面风险评估。",
        "",
        f"审计申请：{contact}",
    ]
    return "\n".join(lines)


def _metric_list_en(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"- Models tested: {summary['n_models']}",
            f"- Prompt-temperature configs: {summary['n_configs']}",
            f"- Samples judged: {summary['n_samples']}",
            f"- Empirical failure probability: {_pct(summary['empirical_failure_probability'])}",
            f"- APST reliability score: {_pct(summary['apst_score'])}",
            f"- Repeated-use risk@{summary['risk_horizon']}: "
            f"{_pct(summary['repeated_inference_risk'])}",
        ]
    )


def _metric_list_zh(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"- 测试模型数：{summary['n_models']}",
            f"- 提示词-温度配置数：{summary['n_configs']}",
            f"- 已评审样本数：{summary['n_samples']}",
            f"- 经验失败概率：{_pct(summary['empirical_failure_probability'])}",
            f"- APST 可靠性分数：{_pct(summary['apst_score'])}",
            f"- 重复使用风险@{summary['risk_horizon']}："
            f"{_pct(summary['repeated_inference_risk'])}",
        ]
    )


def _model_table(models: list[dict[str, Any]], *, horizon: int) -> str:
    if not models:
        return "_No result rows found._"

    lines = [
        "| Model | Configs | Samples | Failures | Failure probability | APST score | "
        f"Risk@{horizon} |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in models:
        lines.append(
            "| {model} | {configs} | {samples} | {failures} | {failure_prob} | {score} | "
            "{risk} |".format(
                model=item["model"],
                configs=item["n_configs"],
                samples=item["n_samples"],
                failures=item["n_failures"],
                failure_prob=_pct(item["empirical_failure_probability"]),
                score=_pct(item["apst_score"]),
                risk=_pct(item["repeated_inference_risk"]),
            )
        )
    return "\n".join(lines)


def _failure_mode_table(failure_modes: list[tuple[str, int]]) -> str:
    if not failure_modes:
        return "_No failure-mode counts found._"
    lines = ["| Label | Count |", "| --- | ---: |"]
    for label, count in failure_modes:
        lines.append(f"| {label} | {count} |")
    return "\n".join(lines)


def _risk_horizon(rows: list[dict[str, Any]]) -> int:
    for row in rows:
        value = row.get("apst_risk_horizon")
        if value is not None:
            return max(1, _as_int(value))
    return 10


def _default_report_path(results_path: Path, lang: str) -> Path:
    return results_path.with_name(f"{results_path.stem}_report_{lang}.md")


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _as_int(value: Any) -> int:
    if value is None:
        return 0
    return int(float(value))


def _as_float(value: Any) -> float:
    if value is None:
        return 0.0
    return float(value)
