from __future__ import annotations

import json
import os
from typing import Protocol

from openai import OpenAI

from prompt_from_examples.models import SupportDataset, TaskDefinition

HEURISTIC_PRIORITY_POLICY = (
    "Use high priority for blocked access, duplicate charges or refund disputes, "
    "and crashes or broken existing features. "
    "Use medium priority for ownership transfers, card or renewal updates, and "
    "sorting or import issues that degrade work without fully blocking it. "
    "Use low priority for feature requests and non-urgent billing corrections such as "
    "company-name or tax-number updates."
)


def _build_instruction(
    *,
    raw_user_intent: str | None,
    output_contract: dict[str, list[str]],
) -> str:
    if raw_user_intent:
        cleaned_intent = " ".join(raw_user_intent.split())
        lead = cleaned_intent.rstrip(".")
    else:
        lead = "Classify each incoming support request into JSON only"

    return (
        f"{lead}. Return JSON only with keys category, priority, and language. "
        f"Use category one of {', '.join(output_contract['category'])}. "
        f"Use priority one of {', '.join(output_contract['priority'])}. "
        f"Use language one of {', '.join(output_contract['language'])}. "
        f"{HEURISTIC_PRIORITY_POLICY}"
    )


class TaskInferenceEngine(Protocol):
    def infer_task(
        self, dataset: SupportDataset, raw_user_intent: str | None
    ) -> TaskDefinition: ...


class HeuristicTaskInferenceEngine:
    def infer_task(self, dataset: SupportDataset, raw_user_intent: str | None) -> TaskDefinition:
        output_contract = {
            "category": sorted({example.output.category for example in dataset.examples}),
            "priority": sorted({example.output.priority for example in dataset.examples}),
            "language": sorted({example.output.language for example in dataset.examples}),
        }

        instruction = _build_instruction(
            raw_user_intent=raw_user_intent,
            output_contract=output_contract,
        )

        if raw_user_intent:
            rationale = (
                "Started from the raw user intent, then added the output contract and a "
                "dataset-derived priority policy."
            )
        else:
            rationale = (
                "No user prompt was provided, so the task and its priority policy were inferred "
                "from the repeated output shape and recurring patterns in the dataset."
            )

        return TaskDefinition(
            instruction=instruction,
            rationale=rationale,
            output_contract=output_contract,
        )


class OpenAITaskInferenceEngine:
    def __init__(self, model: str | None = None) -> None:
        self._fallback = HeuristicTaskInferenceEngine()
        self._client = OpenAI()
        self._model = model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

    def infer_task(self, dataset: SupportDataset, raw_user_intent: str | None) -> TaskDefinition:
        seed_definition = self._fallback.infer_task(dataset, raw_user_intent)
        sampled_examples = dataset.examples[: min(len(dataset.examples), 8)]
        example_payload = [
            {
                "input": example.input,
                "output": example.output.model_dump(),
            }
            for example in sampled_examples
        ]
        user_intent = raw_user_intent or "No explicit user prompt was provided."
        response = self._client.responses.create(
            model=self._model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You turn input/output datasets into concise prompt instructions. "
                        "Preserve the schema, stay general, capture priority policy, "
                        "and do not overfit to the sample."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"User intent: {user_intent}\n"
                        f"Seed instruction: {seed_definition.instruction}\n"
                        f"Output contract: {json.dumps(seed_definition.output_contract)}\n"
                        f"Examples: {json.dumps(example_payload, ensure_ascii=False)}\n"
                        "Write one concise instruction paragraph that includes useful "
                        "priority guidance when the examples support it."
                    ),
                },
            ],
        )
        refined_instruction = response.output_text.strip() or seed_definition.instruction
        return TaskDefinition(
            instruction=refined_instruction,
            rationale="OpenAI refined the seed instruction that was inferred from the dataset.",
            output_contract=seed_definition.output_contract,
        )


def build_task_inference_engine(use_openai: bool) -> TaskInferenceEngine:
    if use_openai:
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is required when --use-openai is enabled.")
        return OpenAITaskInferenceEngine()
    return HeuristicTaskInferenceEngine()
