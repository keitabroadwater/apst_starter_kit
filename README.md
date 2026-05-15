# APST Starter Kit

APST stands for **Accelerated Prompt Stress Testing**. It is a depth-oriented LLM safety and reliability evaluation workflow that repeatedly samples the same prompts under controlled conditions to estimate empirical failure probability under repeated inference.

This v0.1 starter kit is local-first. You can run the mock demo without an API key, then swap in OpenAI, Together.ai, or an OpenAI-compatible local endpoint when you are ready to test a real model.

## Research Background

This starter kit is based on the Accelerated Prompt Stress Testing (APST) research introduced by Keita Broadwater.

APST is a depth-oriented evaluation framework for LLM safety and reliability. Instead of evaluating a model once per prompt, APST repeatedly samples identical or near-identical prompts under controlled operational conditions, then estimates empirical failure probability under repeated inference.

This helps surface deployment-relevant risks such as refusal inconsistency, unsafe completions, hallucinations, and reliability gaps that may be hidden by shallow or low-sample benchmark evaluation.

The starter kit is intended to make the APST workflow easier to try locally:

- repeated prompt sampling;
- judging;
- empirical failure-rate estimation;
- operational risk projection;
- English, Chinese, and bilingual report generation.

## Related Papers

### Evaluating LLM Safety Under Repeated Inference via Accelerated Prompt Stress Testing

This is the primary APST paper. It introduces Accelerated Prompt Stress Testing as a depth-oriented framework for evaluating LLM safety under repeated inference. APST repeatedly samples prompts under controlled conditions, such as decoding temperature, and models observed failures as stochastic outcomes using Bernoulli and binomial formulations. The paper shows that models with similar shallow-evaluation scores can exhibit meaningfully different empirical failure rates under repeated sampling.

- arXiv: https://arxiv.org/abs/2602.11786
- DOI: https://doi.org/10.48550/arXiv.2602.11786

### Evaluating Reliability Gaps in Large Language Model Safety via Repeated Prompt Sampling

This conference paper presents APST through the lens of reliability gaps in LLM safety evaluation and was accepted at CCAI 2026. It contrasts breadth-oriented safety benchmarks with depth-oriented repeated sampling and emphasizes operational failures that emerge under repeated use.

- arXiv: https://arxiv.org/abs/2604.09606
- DOI: https://doi.org/10.48550/arXiv.2604.09606

## How This Starter Kit Relates to the Papers

The APST Starter Kit is a practical, local-first implementation inspired by the APST papers above. It is designed for researchers, practitioners, and enterprise AI teams who want to experiment with the core APST workflow on their own machines.

This repository is not a full reproduction package for every experiment in the papers. Instead, it provides a lightweight starting point for running APST-style evaluations locally, generating reports, and adapting the workflow to research, enterprise, or deployment-readiness contexts.

For full-scale APST studies, users should carefully control prompt selection, model configuration, judge configuration, sample depth, temperature, language, and reporting assumptions.

## Quickstart

Install from PyPI after the first release:

```bash
pip install apst-starter-kit
apst init my-apst-demo
cd my-apst-demo

apst run --config configs/demo_mock.yaml
apst report --results outputs/demo_results.csv --lang both
```

Or run from a Git checkout:

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

Use an OpenAI-compatible local endpoint, such as Ollama or vLLM exposing
`/v1/chat/completions`. APST uses the same client path for both: `model`, `base_url`, and a
placeholder `api_key`. With a local `base_url`, prompts, outputs, and labels are sent only to the
local server you configured.

```yaml
models:
  - model: llama3.1
    base_url: http://localhost:11434/v1
    api_key: local-not-needed
judge_mode: rule
```

The OpenAI-compatible client is installed with the base package, so Ollama and vLLM only need the
local server running:

```bash
apst run --config configs/ollama_local.yaml
apst run --config configs/vllm_local.yaml
```

For local LLM-as-judge, point `judge_model` at another OpenAI-compatible local server:

```yaml
judge_mode: llm
judge_model:
  model: llama3.1
  base_url: http://localhost:11434/v1
  api_key: local-not-needed
```

See `docs/local_model_servers.md` for Ollama and vLLM server commands.

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

## Publishing

See `docs/publishing.md` for GitHub release and PyPI publishing steps. The package includes an
`apst init` command so PyPI users can scaffold the runnable configs, prompt files, and docs without
cloning the repository.
