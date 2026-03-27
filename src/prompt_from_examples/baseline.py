from __future__ import annotations

from prompt_from_examples.models import OptimizedPrompt, SupportDataset, TaskDefinition
from prompt_from_examples.prompts import render_prompt_document

BASELINE_INSTRUCTION = (
    "Classify each incoming support request into JSON. Use category one of billing, bug, "
    "feature_request, account_access. Use high priority for blocked access, broken existing "
    "features, or urgent money problems. Use medium priority for important but not fully "
    "blocking billing or access work. Use low priority for suggestions and nice-to-have "
    "requests. Use language tr for Turkish and en for English."
)

BASELINE_OUTPUT_CONTRACT = {
    "category": ["billing", "bug", "feature_request", "account_access"],
    "priority": ["low", "medium", "high"],
    "language": ["tr", "en"],
}


def build_baseline_prompt(dataset: SupportDataset) -> str:
    prompt = OptimizedPrompt(
        task=TaskDefinition(
            instruction=BASELINE_INSTRUCTION,
            rationale="Hand-authored baseline prompt for the initial few-shot examples.",
            output_contract=BASELINE_OUTPUT_CONTRACT,
        ),
        examples=dataset.examples,
    )
    return render_prompt_document(prompt)
