from __future__ import annotations

import re
from collections import defaultdict
from difflib import SequenceMatcher

from prompt_from_examples.models import SupportDataset, SupportExample

CATEGORY_HINTS: dict[str, set[str]] = {
    "billing": {
        "billing",
        "invoice",
        "refund",
        "charged",
        "charge",
        "payment",
        "card",
        "renewal",
        "subscription",
        "fatura",
        "kart",
    },
    "bug": {
        "bug",
        "broken",
        "error",
        "issue",
        "problem",
        "crash",
        "fails",
        "freeze",
        "working",
        "export",
        "report",
        "acilm",
        "kapan",
    },
    "feature_request": {
        "feature",
        "request",
        "add",
        "support",
        "dark",
        "mode",
        "google",
        "sheets",
        "shortcut",
        "template",
        "filtre",
        "eklenebilir",
    },
    "account_access": {
        "login",
        "password",
        "reset",
        "code",
        "account",
        "unlock",
        "access",
        "mail",
        "2fa",
        "verification",
        "sifre",
        "hesap",
    },
}

PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}
NON_QUERY_COVERAGE_TARGETS = [
    ("billing", "high"),
    ("account_access", "high"),
    ("bug", "high"),
    ("feature_request", "low"),
    ("billing", "low"),
    ("account_access", "medium"),
    ("bug", "medium"),
    ("billing", "medium"),
]


def tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"\w+", text.lower()) if token}


def infer_focus_categories(query: str | None) -> list[str]:
    if not query:
        return []

    query_tokens = tokenize(query)
    matched_categories: list[str] = []
    for category, hints in CATEGORY_HINTS.items():
        if query_tokens & hints:
            matched_categories.append(category)
    return matched_categories


def score_example(query: str | None, example: SupportExample) -> float:
    if not query:
        return 0.0

    query_tokens = tokenize(query)
    example_tokens = tokenize(example.input)
    overlap = len(query_tokens & example_tokens) / max(len(query_tokens), 1)
    ratio = SequenceMatcher(None, query.lower(), example.input.lower()).ratio()
    category_bonus = 0.0
    for token in CATEGORY_HINTS[example.output.category]:
        if token in query_tokens:
            category_bonus += 0.1
    priority_bonus = 0.2 if example.output.priority in query_tokens else 0.0
    language_bonus = 0.2 if example.output.language in query_tokens else 0.0
    return overlap + ratio + category_bonus + priority_bonus + language_bonus


def _representative_examples_by_category(dataset: SupportDataset) -> list[SupportExample]:
    grouped: dict[str, list[SupportExample]] = defaultdict(list)
    for example in dataset.examples:
        grouped[example.output.category].append(example)

    representatives: list[SupportExample] = []
    for category in sorted(grouped):
        ranked = sorted(
            grouped[category],
            key=lambda example: (PRIORITY_ORDER[example.output.priority], len(example.input)),
        )
        representatives.append(ranked[0])
    return representatives


def _best_example_for_category(
    dataset: SupportDataset,
    category: str,
    query: str | None,
) -> SupportExample | None:
    candidates = [
        example for example in dataset.examples if example.output.category == category
    ]
    if not candidates:
        return None

    if query:
        ranked = sorted(
            candidates,
            key=lambda example: (
                score_example(query, example),
                -PRIORITY_ORDER[example.output.priority],
            ),
            reverse=True,
        )
        return ranked[0]

    ranked = sorted(
        candidates,
        key=lambda example: (PRIORITY_ORDER[example.output.priority], len(example.input)),
    )
    return ranked[0]


def _best_example_for_pair(
    dataset: SupportDataset,
    category: str,
    priority: str,
    query: str | None,
) -> SupportExample | None:
    candidates = [
        example
        for example in dataset.examples
        if example.output.category == category and example.output.priority == priority
    ]
    if not candidates:
        return None

    if query:
        ranked = sorted(
            candidates,
            key=lambda example: (
                score_example(query, example),
                -len(tokenize(example.input)),
            ),
            reverse=True,
        )
        return ranked[0]

    ranked = sorted(candidates, key=lambda example: len(example.input), reverse=True)
    return ranked[0]


def select_examples(
    dataset: SupportDataset,
    query: str | None,
    max_examples: int = 6,
) -> list[SupportExample]:
    if max_examples <= 0:
        raise ValueError("max_examples must be greater than zero.")

    selected: list[SupportExample] = []
    selected_inputs: set[str] = set()
    selected_pairs: set[tuple[str, str]] = set()

    focus_categories = infer_focus_categories(query)
    if focus_categories:
        for category in focus_categories:
            best = _best_example_for_category(dataset, category, query)
            if best:
                selected.append(best)
                selected_inputs.add(best.input)
                selected_pairs.add((best.output.category, best.output.priority))

    if not query:
        for example in _representative_examples_by_category(dataset):
            if example.input not in selected_inputs and len(selected) < max_examples:
                selected.append(example)
                selected_inputs.add(example.input)
                selected_pairs.add((example.output.category, example.output.priority))

        for category, priority in NON_QUERY_COVERAGE_TARGETS:
            if len(selected) >= max_examples:
                break
            if (category, priority) in selected_pairs:
                continue
            best = _best_example_for_pair(dataset, category, priority, query=None)
            if best and best.input not in selected_inputs:
                selected.append(best)
                selected_inputs.add(best.input)
                selected_pairs.add((best.output.category, best.output.priority))
    else:
        all_categories = sorted({example.output.category for example in dataset.examples})
        if max_examples >= len(all_categories):
            covered_categories = {example.output.category for example in selected}
            missing_categories = [
                category for category in all_categories if category not in covered_categories
            ]
            for category in missing_categories:
                if len(selected) >= max_examples:
                    break
                best = _best_example_for_category(dataset, category, query)
                if best and best.input not in selected_inputs:
                    selected.append(best)
                    selected_inputs.add(best.input)
                    selected_pairs.add((best.output.category, best.output.priority))

        for category, priority in NON_QUERY_COVERAGE_TARGETS:
            if len(selected) >= max_examples:
                break
            if (category, priority) in selected_pairs:
                continue
            best = _best_example_for_pair(dataset, category, priority, query)
            if best and best.input not in selected_inputs:
                selected.append(best)
                selected_inputs.add(best.input)
                selected_pairs.add((best.output.category, best.output.priority))

    remaining = sorted(
        dataset.examples,
        key=lambda example: (
            score_example(query, example),
            -len(tokenize(example.input)),
        ),
        reverse=True,
    )
    for example in remaining:
        if len(selected) >= max_examples:
            break
        if example.input in selected_inputs:
            continue
        selected.append(example)
        selected_inputs.add(example.input)
        selected_pairs.add((example.output.category, example.output.priority))

    return selected[:max_examples]
