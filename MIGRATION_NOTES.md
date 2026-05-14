# Migration Notes

This clean project keeps the reusable evaluation path from the original repo and leaves behind experiment-specific clutter.

## Kept

- `PromptSpec` / prompt registry concepts
- AIRBench prompt freezing and AIRBench category metadata mapping
- OpenAI and Together.ai model runners
- rate limiting for Together.ai calls
- repeated sampling for APST-style reliability measurement
- LLM-as-judge response classification
- reliability metrics and model-level aggregation
- AIRBench-style one-shot scoring
- JSON checkpoints and final result files

## Simplified

- The old phase-specific runners are replaced by one APST runner in `src/llm_eval/experiments/apst.py`.
- Complex Phase 1 / Phase 3A / Phase 3B checkpoint semantics are replaced by one config-level checkpoint key: model, prompt, temperature, and sample count.
- Generated API errors are explicit `generation_error` classifications instead of being silently mixed into empty responses.
- CLI behavior is consolidated in `src/llm_eval/cli.py`.

## Dropped

- `zindi_competition/`
- notebooks and generated figures
- archived experiment outputs
- one-off phase scripts
- stale-import helper scripts
- broad planning notes from previous experiment phases

## Original To Clean Mapping

| Original area | Clean equivalent |
| --- | --- |
| `apst/prompts.py` | `src/llm_eval/prompts.py` |
| `apst/data/airbench_loader.py` | `src/llm_eval/datasets/airbench.py` |
| `apst/models/*` | `src/llm_eval/models/*` |
| `apst/generation/*` | `src/llm_eval/generation/*` |
| `apst/scoring/*` | `src/llm_eval/scoring/*` |
| `apst/experiments/run_apst_experiment.py` | `src/llm_eval/experiments/apst.py` |
| `apst/experiments/run_airbench_evaluation.py` | `src/llm_eval/experiments/airbench.py` |
| `scripts/pull_airbench.py` | `scripts/freeze_airbench.py` and `llm-eval freeze-airbench` |

