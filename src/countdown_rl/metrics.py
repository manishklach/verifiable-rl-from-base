"""Metrics and failure taxonomy independent of model inference."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Any


def classify_failure(record: dict[str, Any]) -> str:
    if record.get("correct"):
        return "correct"
    if not record.get("has_answer_tag"):
        return "missing_answer_tag"
    if not record.get("parseable"):
        return "invalid_expression"
    if not record.get("uses_numbers_exactly_once"):
        return "wrong_number_usage"
    return "wrong_value"


def pass_at_k(records: Iterable[dict[str, Any]], k: int) -> float:
    grouped: dict[str, list[bool]] = defaultdict(list)
    for row in records:
        grouped[str(row["problem_id"])].append(bool(row["correct"]))
    if not grouped:
        return 0.0
    return sum(any(values[:k]) for values in grouped.values()) / len(grouped)


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot summarize zero records")
    problem_count = len({row["problem_id"] for row in records})
    failures = Counter(classify_failure(row) for row in records)
    by_arity: dict[str, list[bool]] = defaultdict(list)
    by_difficulty: dict[str, list[bool]] = defaultdict(list)
    for row in records:
        by_arity[str(len(row["nums"]))].append(bool(row["correct"]))
        if row.get("difficulty_band"):
            by_difficulty[str(row["difficulty_band"])].append(bool(row["correct"]))
    max_k = max(Counter(row["problem_id"] for row in records).values())
    return {
        "problems": problem_count,
        "completions": len(records),
        "strict_accuracy": sum(row["correct"] for row in records) / len(records),
        "parseable_rate": sum(row["parseable"] for row in records) / len(records),
        "exact_number_usage_rate": sum(row["uses_numbers_exactly_once"] for row in records)
        / len(records),
        "format_rate": sum(row["has_answer_tag"] for row in records) / len(records),
        "pass_at_k": {str(k): pass_at_k(records, k) for k in sorted({1, max_k})},
        "accuracy_by_arity": {
            key: sum(values) / len(values) for key, values in sorted(by_arity.items())
        },
        "accuracy_by_difficulty": {
            key: sum(values) / len(values) for key, values in sorted(by_difficulty.items())
        },
        "failure_taxonomy": dict(sorted(failures.items())),
        "mean_completion_characters": sum(len(row.get("completion", "")) for row in records)
        / len(records),
    }

