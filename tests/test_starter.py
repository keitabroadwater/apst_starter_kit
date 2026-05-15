from pathlib import Path

import pytest

from llm_eval.starter import init_starter_workspace


def test_init_starter_workspace_copies_runnable_assets(tmp_path: Path):
    target = init_starter_workspace(tmp_path / "demo")

    assert (target / "README.md").exists()
    assert (target / "configs" / "demo_mock.yaml").exists()
    assert (target / "configs" / "ollama_local.yaml").exists()
    assert (target / "data" / "prompts" / "demo_prompts.json").exists()
    assert (target / "docs" / "local_model_servers.md").exists()


def test_init_starter_workspace_refuses_to_overwrite_without_force(tmp_path: Path):
    target = init_starter_workspace(tmp_path / "demo")

    with pytest.raises(FileExistsError):
        init_starter_workspace(target)

    init_starter_workspace(target, force=True)
