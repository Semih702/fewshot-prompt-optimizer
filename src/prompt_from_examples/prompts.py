from __future__ import annotations

import json

from prompt_from_examples.models import OptimizedPrompt, SupportExample


def format_output(example: SupportExample) -> str:
    return json.dumps(example.output.model_dump(), ensure_ascii=False, sort_keys=True)


def render_example_block(example: SupportExample) -> str:
    return f"Input: {example.input}\nOutput: {format_output(example)}"


def render_prompt_document(prompt: OptimizedPrompt) -> str:
    contract_lines = [
        f"- {field}: {', '.join(values)}" for field, values in prompt.task.output_contract.items()
    ]
    examples_block = "\n\n".join(render_example_block(example) for example in prompt.examples)

    return "\n".join(
        [
            "You are a careful support triage assistant.",
            "",
            "Task:",
            prompt.task.instruction,
            "",
            "Output contract:",
            "- Return JSON only.",
            "- Keys must be exactly: category, priority, language.",
            *contract_lines,
            "",
            "Few-shot examples:",
            examples_block,
            "",
            "Now classify the next message.",
            "Input: {user_input}",
            "Output:",
        ]
    )
