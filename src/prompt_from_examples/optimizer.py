from __future__ import annotations

from prompt_from_examples.models import OptimizedPrompt, SupportDataset
from prompt_from_examples.selection import select_examples
from prompt_from_examples.task_inference import TaskInferenceEngine


class PromptOptimizer:
    def __init__(self, inference_engine: TaskInferenceEngine) -> None:
        self._inference_engine = inference_engine

    def build_prompt(
        self,
        dataset: SupportDataset,
        raw_user_intent: str | None,
        max_examples: int = 6,
    ) -> OptimizedPrompt:
        task = self._inference_engine.infer_task(dataset, raw_user_intent)
        examples = select_examples(
            dataset=dataset,
            query=raw_user_intent,
            max_examples=max_examples,
        )
        return OptimizedPrompt(task=task, examples=examples)
