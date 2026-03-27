from pathlib import Path

from prompt_from_examples.baseline import build_baseline_prompt
from prompt_from_examples.datasets import load_dataset
from prompt_from_examples.prompts import format_output

SMALL_DATASET = Path("prompt_datasets/support_triage_fewshot_small.json")


def test_baseline_prompt_contains_contract() -> None:
    prompt = build_baseline_prompt(load_dataset(SMALL_DATASET))

    assert "Return JSON only." in prompt
    assert "category, priority, language" in prompt
    assert "billing" in prompt
    assert "bug" in prompt
    assert "feature_request" in prompt
    assert "account_access" in prompt


def test_baseline_prompt_renders_every_small_example() -> None:
    dataset = load_dataset(SMALL_DATASET)
    prompt = build_baseline_prompt(dataset)

    for example in dataset.examples:
        assert example.input in prompt
        assert format_output(example) in prompt
