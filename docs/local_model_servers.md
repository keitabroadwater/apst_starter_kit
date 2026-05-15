# Local Model Servers

APST talks to local models through OpenAI-compatible HTTP servers. It does not include or require
an APST-hosted model server. If `base_url` points at localhost, generation and local LLM judging
stay on the machine or network endpoint you chose.

## Config Shape

Local configs can use the minimal OpenAI-compatible shape:

```yaml
models:
  - model: llama3.1
    base_url: http://localhost:11434/v1
    api_key: local-not-needed
```

When `base_url` is set, APST defaults the provider to `openai_compatible`. You can also write
`provider: openai_compatible`, `provider: ollama`, or `provider: vllm`; all use the same
OpenAI-compatible client abstraction.

## Ollama

Ollama is the easiest local path for laptops and demos.

```bash
ollama pull llama3.1
ollama serve
```

In another shell:

```bash
apst run --config configs/ollama_local.yaml
```

The Ollama config uses:

```yaml
models:
  - model: llama3.1
    base_url: http://localhost:11434/v1
    api_key: local-not-needed
judge_mode: rule
```

`judge_mode: rule` keeps judging deterministic and local. To use a local LLM judge, add a
`judge_model` with the same `base_url` shape:

```yaml
judge_mode: llm
judge_model:
  model: llama3.1
  base_url: http://localhost:11434/v1
  api_key: local-not-needed
```

## vLLM

vLLM is the serious inference path for GPUs, batching, and higher-throughput APST runs.

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000
```

In another shell:

```bash
apst run --config configs/vllm_local.yaml
```

The vLLM config uses:

```yaml
models:
  - model: meta-llama/Llama-3.1-8B-Instruct
    base_url: http://localhost:8000/v1
    api_key: local-not-needed
judge_mode: rule
```

For larger local audits, increase `n_samples_per_prompt`, `max_prompts`, and `max_workers` after
confirming the model server is stable.
