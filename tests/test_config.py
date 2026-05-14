from llm_eval.config import load_config


def test_load_demo_mock_config():
    config = load_config("configs/demo_mock.yaml")

    assert config.models[0].provider == "mock"
    assert config.judge_mode == "rule"
    assert config.output_dir == "outputs"
    assert config.result_name == "demo_results"
