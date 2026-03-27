from pathlib import Path

from prompt_from_examples.datasets import load_dataset
from prompt_from_examples.task_inference import HeuristicTaskInferenceEngine

LARGE_DATASET = Path("prompt_datasets/support_triage_dataset.json")


def test_heuristic_task_inference_includes_priority_policy() -> None:
    task = HeuristicTaskInferenceEngine().infer_task(
        dataset=load_dataset(LARGE_DATASET),
        raw_user_intent=None,
    )

    assert "duplicate charges or refund disputes" in task.instruction
    assert "ownership transfers" in task.instruction
    assert "company-name or tax-number updates" in task.instruction


def test_heuristic_task_inference_preserves_user_intent_and_policy() -> None:
    task = HeuristicTaskInferenceEngine().infer_task(
        dataset=load_dataset(LARGE_DATASET),
        raw_user_intent="Turn messy support requests into stable JSON labels for the ops team",
    )

    assert task.instruction.startswith(
        "Turn messy support requests into stable JSON labels for the ops team."
    )
    assert "Use low priority for feature requests" in task.instruction
