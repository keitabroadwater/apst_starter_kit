import pytest

from llm_eval.config import ModelConfig, config_from_dict, load_config
from llm_eval.models.factory import OPENAI_COMPATIBLE_PROVIDERS, create_model_runner


def test_load_demo_mock_config():
    config = load_config("configs/demo_mock.yaml")

    assert config.models[0].provider == "mock"
    assert config.judge_mode == "rule"
    assert config.output_dir == "outputs"
    assert config.result_name == "demo_results"


def test_load_ollama_config_defaults_to_openai_compatible():
    config = load_config("configs/ollama_local.yaml")

    model = config.models[0]
    assert model.name == model.model_id
    assert model.model_id
    assert model.provider == "openai_compatible"
    assert model.base_url == "http://localhost:11434/v1"
    assert model.api_key == "local-not-needed"
    assert config.judge_mode == "rule"


def test_load_vllm_config_uses_same_openai_compatible_shape():
    config = load_config("configs/vllm_local.yaml")

    model = config.models[0]
    assert model.name == "meta-llama/Llama-3.1-8B-Instruct"
    assert model.model_id == "meta-llama/Llama-3.1-8B-Instruct"
    assert model.provider == "openai_compatible"
    assert model.base_url == "http://localhost:8000/v1"
    assert model.api_key == "local-not-needed"


def test_local_provider_aliases_route_to_openai_compatible_runner():
    assert "ollama" in OPENAI_COMPATIBLE_PROVIDERS
    assert "vllm" in OPENAI_COMPATIBLE_PROVIDERS


def test_local_llm_judge_uses_same_minimal_config_shape():
    config = config_from_dict(
        {
            "models": [
                {
                    "model": "llama3.1",
                    "base_url": "http://localhost:11434/v1",
                    "api_key": "local-not-needed",
                }
            ],
            "judge_mode": "llm",
            "judge_model": {
                "model": "qwen2.5:7b",
                "base_url": "http://localhost:11434/v1",
                "api_key": "local-not-needed",
            },
        }
    )

    assert config.models[0].provider == "openai_compatible"
    assert config.judge_model is not None
    assert config.judge_model.model_id == "qwen2.5:7b"
    assert config.judge_model.provider == "openai_compatible"


def test_local_provider_alias_requires_base_url():
    config = ModelConfig(name="llama3.1", model_id="llama3.1", provider="ollama")

    with pytest.raises(ValueError, match="requires base_url"):
        create_model_runner(config)
