from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv

from prompt_from_examples.baseline import build_baseline_prompt
from prompt_from_examples.datasets import load_dataset
from prompt_from_examples.evals import (
    build_developer_baseline_variant,
    build_production_generated_variant,
    build_prompt_executor,
    compare_variants,
    format_comparison_report,
    format_failures,
)
from prompt_from_examples.optimizer import PromptOptimizer
from prompt_from_examples.prompts import render_prompt_document
from prompt_from_examples.task_inference import build_task_inference_engine

app = typer.Typer(no_args_is_help=True, help="Build few-shot prompts from datasets.")

DEFAULT_SMALL_DATASET = Path("prompt_datasets/support_triage_fewshot_small.json")
DEFAULT_EVAL_DATASET = Path("prompt_datasets/support_triage_eval.json")


def _default_large_dataset() -> Path:
    return Path(
        os.getenv(
            "PROMPT_FROM_EXAMPLES_DEFAULT_DATASET",
            "prompt_datasets/support_triage_dataset.json",
        )
    )


@app.command("baseline")
def baseline(dataset: Path = DEFAULT_SMALL_DATASET) -> None:
    """Render the small hand-authored prompt."""
    prompt = build_baseline_prompt(load_dataset(dataset))
    typer.echo(prompt)


@app.command("optimize")
def optimize(
    dataset: Annotated[
        Path | None,
        typer.Option("--dataset", help="Training dataset used for prompt generation."),
    ] = None,
    user_intent: Annotated[
        str | None,
        typer.Option("--user-intent", help="A rough human description of the task."),
    ] = None,
    max_examples: Annotated[int, typer.Option(min=1, max=12)] = 6,
    use_openai: Annotated[
        bool,
        typer.Option(
            "--use-openai",
            help="Use OpenAI to refine the task instruction when OPENAI_API_KEY is set.",
        ),
    ] = False,
) -> None:
    """Build a prompt from the larger dataset."""
    load_dotenv()
    resolved_dataset = dataset or _default_large_dataset()
    inference_engine = build_task_inference_engine(use_openai=use_openai)
    optimizer = PromptOptimizer(inference_engine)
    optimized_prompt = optimizer.build_prompt(
        dataset=load_dataset(resolved_dataset),
        raw_user_intent=user_intent,
        max_examples=max_examples,
    )
    typer.echo(render_prompt_document(optimized_prompt))
    typer.echo("")
    typer.echo(f"Selection rationale: {optimized_prompt.task.rationale}")
    typer.echo("Selected examples:")
    for example in optimized_prompt.examples:
        typer.echo(
            f"- {example.output.category} / {example.output.priority} / "
            f"{example.output.language}: {example.input}"
        )


@app.command("compare")
def compare(
    baseline_dataset: Annotated[
        Path,
        typer.Option(
            "--baseline-dataset",
            help="Small developer-authored few-shot dataset for prompt A.",
        ),
    ] = DEFAULT_SMALL_DATASET,
    train_dataset: Annotated[
        Path | None,
        typer.Option(
            "--train-dataset",
            help="Dataset used to generate production prompt B.",
        ),
    ] = None,
    eval_dataset: Annotated[
        Path,
        typer.Option(
            "--eval-dataset",
            help="Held-out evaluation dataset used to compare prompt quality.",
        ),
    ] = DEFAULT_EVAL_DATASET,
    user_intent: Annotated[
        str | None,
        typer.Option(
            "--user-intent",
            help="Optional rough human intent for generating prompt B.",
        ),
    ] = None,
    max_examples: Annotated[int, typer.Option(min=1, max=12)] = 6,
    use_openai_inference: Annotated[
        bool,
        typer.Option(
            "--use-openai-inference",
            help="Use OpenAI while generating prompt B itself.",
        ),
    ] = False,
    eval_backend: Annotated[
        str,
        typer.Option(
            "--eval-backend",
            help="Which executor to use for evaluation: simulated or openai.",
        ),
    ] = "simulated",
    show_failures: Annotated[
        bool,
        typer.Option(
            "--show-failures",
            help="Print per-example failures for both prompt variants.",
        ),
    ] = False,
) -> None:
    """Compare prompt A and prompt B on the same held-out eval set."""
    load_dotenv()
    resolved_train_dataset = train_dataset or _default_large_dataset()
    normalized_backend = eval_backend.lower()
    if normalized_backend not in {"simulated", "openai"}:
        raise typer.BadParameter("eval-backend must be either 'simulated' or 'openai'.")

    baseline_variant = build_developer_baseline_variant(load_dataset(baseline_dataset))
    optimized_variant = build_production_generated_variant(
        dataset=load_dataset(resolved_train_dataset),
        raw_user_intent=user_intent,
        max_examples=max_examples,
        inference_engine=build_task_inference_engine(use_openai=use_openai_inference),
    )
    comparison = compare_variants(
        baseline_variant=baseline_variant,
        optimized_variant=optimized_variant,
        eval_dataset=load_dataset(eval_dataset),
        executor=build_prompt_executor(normalized_backend),
    )

    typer.echo(format_comparison_report(comparison))
    if show_failures:
        typer.echo("")
        typer.echo(format_failures(comparison.baseline))
        typer.echo("")
        typer.echo(format_failures(comparison.optimized))


if __name__ == "__main__":
    app()
