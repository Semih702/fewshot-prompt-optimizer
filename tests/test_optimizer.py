from pathlib import Path

from prompt_from_examples.datasets import load_dataset
from prompt_from_examples.optimizer import PromptOptimizer
from prompt_from_examples.prompts import render_prompt_document
from prompt_from_examples.task_inference import HeuristicTaskInferenceEngine

LARGE_DATASET = Path("prompt_datasets/support_triage_dataset.json")


def test_optimizer_infers_instruction_when_user_intent_is_missing() -> None:
    optimizer = PromptOptimizer(HeuristicTaskInferenceEngine())
    optimized_prompt = optimizer.build_prompt(
        dataset=load_dataset(LARGE_DATASET),
        raw_user_intent=None,
        max_examples=5,
    )

    rendered = render_prompt_document(optimized_prompt)

    assert "Classify each incoming support request into JSON only." in rendered
    assert len(optimized_prompt.examples) == 5
    assert "Output contract:" in rendered


def test_optimizer_keeps_the_raw_user_intent_visible() -> None:
    optimizer = PromptOptimizer(HeuristicTaskInferenceEngine())
    raw_intent = "Turn messy support requests into stable JSON labels for the ops team"
    optimized_prompt = optimizer.build_prompt(
        dataset=load_dataset(LARGE_DATASET),
        raw_user_intent=raw_intent,
        max_examples=4,
    )

    assert raw_intent in optimized_prompt.task.instruction
    assert len(optimized_prompt.examples) == 4
