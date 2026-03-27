from pathlib import Path

from prompt_from_examples.datasets import load_dataset
from prompt_from_examples.selection import select_examples

LARGE_DATASET = Path("prompt_datasets/support_triage_dataset.json")


def test_selection_without_query_covers_all_categories() -> None:
    dataset = load_dataset(LARGE_DATASET)
    selected = select_examples(dataset=dataset, query=None, max_examples=4)

    assert {example.output.category for example in selected} == {
        "account_access",
        "billing",
        "bug",
        "feature_request",
    }


def test_selection_with_query_prefers_relevant_categories() -> None:
    dataset = load_dataset(LARGE_DATASET)
    selected = select_examples(
        dataset=dataset,
        query="We need to handle billing disputes and password reset issues",
        max_examples=4,
    )

    categories = {example.output.category for example in selected}
    assert "billing" in categories
    assert "account_access" in categories


def test_selection_with_general_query_preserves_category_coverage() -> None:
    dataset = load_dataset(LARGE_DATASET)
    selected = select_examples(
        dataset=dataset,
        query="Turn messy support requests into stable JSON labels for the ops team",
        max_examples=4,
    )

    assert {example.output.category for example in selected} == {
        "account_access",
        "billing",
        "bug",
        "feature_request",
    }


def test_selection_without_query_adds_priority_diversity_when_budget_allows() -> None:
    dataset = load_dataset(LARGE_DATASET)
    selected = select_examples(dataset=dataset, query=None, max_examples=6)

    selected_pairs = {
        (example.output.category, example.output.priority) for example in selected
    }
    assert ("billing", "low") in selected_pairs
    assert ("bug", "medium") in selected_pairs or ("account_access", "medium") in selected_pairs
