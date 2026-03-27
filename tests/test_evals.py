from pathlib import Path

from prompt_from_examples.datasets import load_dataset
from prompt_from_examples.evals import (
    SimulatedPromptExecutor,
    build_developer_baseline_variant,
    build_production_generated_variant,
    compare_variants,
)
from prompt_from_examples.task_inference import HeuristicTaskInferenceEngine

BASELINE_DATASET = Path("prompt_datasets/support_triage_fewshot_small.json")
TRAIN_DATASET = Path("prompt_datasets/support_triage_dataset.json")
EVAL_DATASET = Path("prompt_datasets/support_triage_eval.json")


def test_generated_prompt_matches_or_beats_baseline_in_simulated_eval() -> None:
    comparison = compare_variants(
        baseline_variant=build_developer_baseline_variant(load_dataset(BASELINE_DATASET)),
        optimized_variant=build_production_generated_variant(
            dataset=load_dataset(TRAIN_DATASET),
            raw_user_intent=None,
            max_examples=6,
            inference_engine=HeuristicTaskInferenceEngine(),
        ),
        eval_dataset=load_dataset(EVAL_DATASET),
        executor=SimulatedPromptExecutor(),
    )

    assert comparison.optimized.exact_match_rate >= comparison.baseline.exact_match_rate
    assert comparison.optimized.exact_match_rate == 1.0
    assert comparison.exact_match_delta >= 0


def test_generated_prompt_reaches_full_priority_accuracy_in_simulated_eval() -> None:
    comparison = compare_variants(
        baseline_variant=build_developer_baseline_variant(load_dataset(BASELINE_DATASET)),
        optimized_variant=build_production_generated_variant(
            dataset=load_dataset(TRAIN_DATASET),
            raw_user_intent="Turn messy support requests into stable JSON labels for the ops team",
            max_examples=6,
            inference_engine=HeuristicTaskInferenceEngine(),
        ),
        eval_dataset=load_dataset(EVAL_DATASET),
        executor=SimulatedPromptExecutor(),
    )

    assert comparison.optimized.field_accuracy["priority"] == 1.0
    assert all(case.exact_match for case in comparison.optimized.case_results)
