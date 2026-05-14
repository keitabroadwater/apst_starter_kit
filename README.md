# APST Starter Kit

APST stands for Accelerated Prompt Stress Testing. It is a depth-oriented LLM safety and
reliability evaluation workflow that repeatedly samples the same prompts under controlled
conditions to estimate empirical failure probability under repeated inference.

This v0.1 starter kit is local-first and conference-friendly. You can run the mock demo without
an API key, then swap in OpenAI, Together.ai, or an OpenAI-compatible local endpoint when you are
ready to test a real model.

## Quickstart

```bash
git clone <repo-url>
cd apst-starter-kit
pip install -e .

apst run --config configs/demo_mock.yaml
apst report --results outputs/demo_results.csv --lang both
```

The demo writes:

- `outputs/demo_results.csv`
- `outputs/demo_results.json`
- `outputs/demo_results_report_both.md`

## What The Demo Does

The mock run:

- loads a small prompt set from `data/prompts/demo_prompts.json`
- repeatedly samples each prompt at two temperatures
- judges every response with the local `rule` judge mode
- computes APST reliability and repeated-use risk metrics
- exports CSV and JSON result files
- generates English, Chinese, or bilingual Markdown reports

No API key or network access is required for `configs/demo_mock.yaml`.

## Configuring Models

Use the mock provider for local demos:

```yaml
models:
  - name: mock-apst-model
    model_id: mock-apst-model
    provider: mock
judge_mode: rule
```

Use OpenAI with an LLM judge:

```bash
export OPENAI_API_KEY=...
apst run --config configs/openai_example.yaml
```

Use an OpenAI-compatible local endpoint, such as a local server exposing `/v1/chat/completions`:

```yaml
models:
  - name: local-openai-compatible
    model_id: llama3.1
    provider: openai_compatible
    base_url: http://localhost:11434/v1
    api_key: local-not-needed
judge_mode: rule
```

Together.ai models are also supported through the existing provider extra:

```bash
pip install -e ".[providers]"
export TOGETHER_API_KEY=...
apst run --config configs/openai_example.yaml --models meta-llama/Llama-3.3-70B-Instruct-Turbo
```

## Judge Modes

- `rule`: local deterministic checks for refusal, harmful operational guidance, crisis-support
  handling, and gibberish. Good for demos, local LLMs, and fast smoke tests.
- `heuristic`: local malformed-output check only. Good when you want a very conservative
  no-network sanity pass.
- `llm`: LLM-as-judge classification using `judge_model`. Good for richer audits and real model
  comparisons.

## APST Metrics

Each prompt/model/temperature config reports:

- `empirical_failure_probability`: observed failures divided by repeated samples
- `reliability`: `1 - empirical_failure_probability`
- `apst_risk_at_10`: estimated chance of at least one failure across 10 independent attempts
- `failure_probability_ci_low` and `failure_probability_ci_high`: Wilson interval bounds
- `failure_mode_distribution`: counts by judge label

The repeated-use estimate is intentionally simple:

```text
Risk@N = 1 - (1 - empirical_failure_probability)^N
```

## Reports

```bash
apst report --results outputs/demo_results.csv --lang en
apst report --results outputs/demo_results.csv --lang zh
apst report --results outputs/demo_results.csv --lang both
```

Use `--audit-contact` to put a real intake link, email address, or conference note in the report.
See `docs/enterprise_audits.md` for the enterprise audit handoff text.

## Legacy Commands

The older `llm-eval` entry point and APST/AIRBench commands are still available:

```bash
llm-eval run-apst --config configs/smoke.yaml
llm-eval freeze-airbench --region us --per-l4 5 --output data/prompts/airbench_us_v1.json
```

## Local Checks

```bash
pytest
```
