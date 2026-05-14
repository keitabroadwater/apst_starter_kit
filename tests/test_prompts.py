from pathlib import Path

from llm_eval.prompts import PromptRegistry, PromptType


def test_load_example_prompt_set():
    registry = PromptRegistry()
    count = registry.load_prompt_set(Path("data/prompts/example_prompts.json"))

    assert count == 3
    prompts = registry.all()
    assert prompts[0].prompt_type == PromptType.SAFETY_HARMFUL
    assert prompts[0].metadata["cate_idx"] == "custom.1"
    assert registry.stats()["total"] == 3

