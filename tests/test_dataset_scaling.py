from pathlib import Path

from prompt_from_examples.datasets import load_dataset

LARGE_DATASET = Path("prompt_datasets/support_triage_dataset.json")
SMALL_DATASET = Path("prompt_datasets/support_triage_fewshot_small.json")


def test_large_dataset_is_bigger_than_small_fewshot_file() -> None:
    small_dataset = load_dataset(SMALL_DATASET)
    large_dataset = load_dataset(LARGE_DATASET)

    assert len(large_dataset.examples) > len(small_dataset.examples)
    assert len(large_dataset.examples) >= 4 * len(small_dataset.examples)
    assert len(large_dataset.examples) >= 24
