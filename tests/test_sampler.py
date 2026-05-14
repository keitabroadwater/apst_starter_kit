from llm_eval.generation.sampler import sample_responses
from llm_eval.models.base import GenerationOutput, ModelRunner


class FakeRunner(ModelRunner):
    def __init__(self):
        super().__init__("fake")
        self.calls = 0

    def generate(self, prompt, *, temperature=1.0, max_tokens=None, **kwargs):
        self.calls += 1
        return GenerationOutput(text=f"{prompt}:{self.calls}", model_name=self.model_name)


def test_sample_responses_preserves_count():
    runner = FakeRunner()
    outputs = sample_responses(
        runner,
        "hello",
        temperature=0.0,
        n_samples=3,
        show_progress=False,
    )

    assert len(outputs) == 3
    assert [output.text for output in outputs] == ["hello:1", "hello:2", "hello:3"]

