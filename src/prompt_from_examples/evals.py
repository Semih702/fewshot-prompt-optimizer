from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Literal, Protocol

from openai import OpenAI

from prompt_from_examples.baseline import build_baseline_prompt
from prompt_from_examples.models import SupportDataset, TriageOutput
from prompt_from_examples.optimizer import PromptOptimizer
from prompt_from_examples.prompts import render_prompt_document
from prompt_from_examples.task_inference import TaskInferenceEngine

FIELD_NAMES = ("category", "priority", "language")
JSON_BLOCK_PATTERN = re.compile(r"\{.*\}", re.DOTALL)
TURKISH_HINTS = {
    "sifre",
    "giris",
    "destek",
    "kisayol",
    "ekleyin",
    "yeniledim",
    "hala",
    "taleplerine",
    "yoneticimize",
    "sahipligini",
}


@dataclass(frozen=True)
class PromptVariant:
    name: str
    role: Literal["developer_baseline", "production_generated"]
    prompt_text: str
    notes: str


@dataclass(frozen=True)
class EvalCaseResult:
    user_input: str
    expected: TriageOutput
    predicted: TriageOutput | None
    raw_output: str
    exact_match: bool
    field_matches: dict[str, bool]
    error: str | None


@dataclass(frozen=True)
class EvalReport:
    variant: PromptVariant
    total_cases: int
    exact_matches: int
    exact_match_rate: float
    field_accuracy: dict[str, float]
    case_results: tuple[EvalCaseResult, ...]


@dataclass(frozen=True)
class ComparisonReport:
    baseline: EvalReport
    optimized: EvalReport
    exact_match_delta: float


class PromptExecutor(Protocol):
    def run_prompt(self, prompt_text: str, user_input: str) -> str:
        ...


def build_developer_baseline_variant(dataset: SupportDataset) -> PromptVariant:
    return PromptVariant(
        name="A / developer baseline",
        role="developer_baseline",
        prompt_text=build_baseline_prompt(dataset),
        notes="Hand-authored prompt used only as the developer baseline.",
    )


def build_production_generated_variant(
    dataset: SupportDataset,
    raw_user_intent: str | None,
    max_examples: int,
    inference_engine: TaskInferenceEngine,
) -> PromptVariant:
    optimizer = PromptOptimizer(inference_engine)
    optimized_prompt = optimizer.build_prompt(dataset, raw_user_intent, max_examples=max_examples)
    return PromptVariant(
        name="B / production generated",
        role="production_generated",
        prompt_text=render_prompt_document(optimized_prompt),
        notes=optimized_prompt.task.rationale,
    )


def _parse_model_output(raw_output: str) -> TriageOutput:
    stripped = raw_output.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        stripped = stripped.replace("json", "", 1).strip()
    match = JSON_BLOCK_PATTERN.search(stripped)
    if not match:
        raise ValueError("Model output did not contain a JSON object.")
    payload = json.loads(match.group(0))
    return TriageOutput.model_validate(payload)


def evaluate_prompt_variant(
    variant: PromptVariant,
    eval_dataset: SupportDataset,
    executor: PromptExecutor,
) -> EvalReport:
    case_results: list[EvalCaseResult] = []
    exact_matches = 0
    field_hits = {field_name: 0 for field_name in FIELD_NAMES}

    for example in eval_dataset.examples:
        raw_output = executor.run_prompt(variant.prompt_text, example.input)
        predicted: TriageOutput | None = None
        field_matches = {field_name: False for field_name in FIELD_NAMES}
        error: str | None = None

        try:
            predicted = _parse_model_output(raw_output)
            field_matches = {
                "category": predicted.category == example.output.category,
                "priority": predicted.priority == example.output.priority,
                "language": predicted.language == example.output.language,
            }
        except (ValueError, json.JSONDecodeError) as exc:
            error = str(exc)

        exact_match = all(field_matches.values())
        if exact_match:
            exact_matches += 1
        for field_name, matched in field_matches.items():
            if matched:
                field_hits[field_name] += 1

        case_results.append(
            EvalCaseResult(
                user_input=example.input,
                expected=example.output,
                predicted=predicted,
                raw_output=raw_output,
                exact_match=exact_match,
                field_matches=field_matches,
                error=error,
            )
        )

    total_cases = len(eval_dataset.examples)
    field_accuracy = {
        field_name: field_hits[field_name] / total_cases for field_name in FIELD_NAMES
    }
    return EvalReport(
        variant=variant,
        total_cases=total_cases,
        exact_matches=exact_matches,
        exact_match_rate=exact_matches / total_cases,
        field_accuracy=field_accuracy,
        case_results=tuple(case_results),
    )


def compare_variants(
    baseline_variant: PromptVariant,
    optimized_variant: PromptVariant,
    eval_dataset: SupportDataset,
    executor: PromptExecutor,
) -> ComparisonReport:
    baseline_report = evaluate_prompt_variant(baseline_variant, eval_dataset, executor)
    optimized_report = evaluate_prompt_variant(optimized_variant, eval_dataset, executor)
    return ComparisonReport(
        baseline=baseline_report,
        optimized=optimized_report,
        exact_match_delta=optimized_report.exact_match_rate - baseline_report.exact_match_rate,
    )


def format_eval_report(report: EvalReport) -> str:
    category_accuracy = report.field_accuracy["category"]
    priority_accuracy = report.field_accuracy["priority"]
    language_accuracy = report.field_accuracy["language"]
    return "\n".join(
        [
            f"{report.variant.name}",
            "  exact match: "
            f"{report.exact_matches}/{report.total_cases} "
            f"({report.exact_match_rate:.1%})",
            f"  category accuracy: {category_accuracy:.1%}",
            f"  priority accuracy: {priority_accuracy:.1%}",
            f"  language accuracy: {language_accuracy:.1%}",
            f"  notes: {report.variant.notes}",
        ]
    )


def format_comparison_report(report: ComparisonReport) -> str:
    delta_label = f"{report.exact_match_delta:+.1%}"
    winner = (
        report.optimized.variant.name
        if report.exact_match_delta > 0
        else report.baseline.variant.name
        if report.exact_match_delta < 0
        else "Tie"
    )
    return "\n\n".join(
        [
            format_eval_report(report.baseline),
            format_eval_report(report.optimized),
            "\n".join(
                [
                    "Comparison",
                    f"  exact-match delta (B - A): {delta_label}",
                    f"  winner: {winner}",
                ]
            ),
        ]
    )


def format_failures(report: EvalReport) -> str:
    lines: list[str] = [f"Failures for {report.variant.name}"]
    failures = [case for case in report.case_results if not case.exact_match]
    if not failures:
        lines.append("  none")
        return "\n".join(lines)

    for case in failures:
        predicted = case.predicted.model_dump() if case.predicted else {"error": case.error}
        lines.extend(
            [
                f"  input: {case.user_input}",
                f"  expected: {case.expected.model_dump()}",
                f"  predicted: {predicted}",
            ]
        )
    return "\n".join(lines)


class SimulatedPromptExecutor:
    def run_prompt(self, prompt_text: str, user_input: str) -> str:
        prediction = TriageOutput(
            category=self._predict_category(user_input),
            priority=self._predict_priority(prompt_text, user_input),
            language=self._predict_language(user_input),
        )
        return json.dumps(prediction.model_dump(), ensure_ascii=False, sort_keys=True)

    def _predict_language(self, user_input: str) -> Literal["tr", "en"]:
        lowered = user_input.lower()
        if re.search(r"[çğıöşü]", lowered):
            return "tr"
        if any(hint in lowered for hint in TURKISH_HINTS):
            return "tr"
        return "en"

    def _predict_category(
        self,
        user_input: str,
    ) -> Literal["billing", "bug", "feature_request", "account_access"]:
        lowered = user_input.lower()

        if any(
            token in lowered
            for token in {
                "add ",
                "add a",
                "option",
                "keyboard shortcut",
                "kisayol",
                "ekleyin",
                "feature",
            }
        ):
            return "feature_request"
        if any(
            token in lowered
            for token in {
                "bill",
                "billed",
                "invoice",
                "charged",
                "charge",
                "payment",
                "refund",
                "renewal",
                "company name",
            }
        ):
            return "billing"
        if any(
            token in lowered
            for token in {
                "password",
                "login",
                "unlock",
                "access",
                "ownership",
                "admin",
                "workspace ownership",
                "sifre",
                "giris",
                "sahipligini",
                "yonetici",
            }
        ):
            return "account_access"
        return "bug"

    def _predict_priority(
        self,
        prompt_text: str,
        user_input: str,
    ) -> Literal["low", "medium", "high"]:
        lowered = user_input.lower()
        category = self._predict_category(user_input)
        has_detailed_priority_policy = all(
            clause in prompt_text.lower()
            for clause in {
                "use high priority for blocked access",
                "use medium priority for important but not fully blocking billing or access work",
                "use low priority for suggestions and nice-to-have requests",
            }
        )
        has_generated_priority_policy = all(
            clause in prompt_text.lower()
            for clause in {
                "duplicate charges or refund disputes",
                "ownership transfers",
                "company-name or tax-number updates",
            }
        )

        if category == "feature_request":
            return "low"

        if not has_detailed_priority_policy and not has_generated_priority_policy:
            if any(
                token in lowered
                for token in {"urgent", "asap", "today", "tomorrow", "demo", "meeting"}
            ):
                return "high"
            return "medium"

        if category == "billing":
            if any(
                token in lowered
                for token in {"charged twice", "billed twice", "duplicate", "refund"}
            ):
                return "high"
            if any(
                token in lowered
                for token in {
                    "company name",
                    "rename",
                    "yanlis gorunuyor",
                    "tax number",
                    "tax id",
                    "legal entity",
                }
            ):
                return "low"
            return "medium"

        if category == "account_access":
            if any(
                token in lowered
                for token in {
                    "giris yapamiyorum",
                    "can't login",
                    "cannot login",
                    "unlock",
                    "before",
                    "maili bana gelmiyor",
                    "hala giris yapamiyorum",
                }
            ):
                return "high"
            return "medium"

        if any(
            token in lowered
            for token in {"crash", "crashes", "freeze", "stopped working", "not working", "broken"}
        ):
            return "high"
        return "medium"


class OpenAIPromptExecutor:
    def __init__(self, model: str | None = None) -> None:
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is required for live OpenAI evaluations.")
        self._client = OpenAI()
        self._model = model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

    def run_prompt(self, prompt_text: str, user_input: str) -> str:
        runtime_prompt = prompt_text.replace("{user_input}", user_input)
        response = self._client.responses.create(model=self._model, input=runtime_prompt)
        return response.output_text.strip()


def build_prompt_executor(
    backend: Literal["simulated", "openai"],
    model: str | None = None,
) -> PromptExecutor:
    if backend == "openai":
        return OpenAIPromptExecutor(model=model)
    return SimulatedPromptExecutor()
