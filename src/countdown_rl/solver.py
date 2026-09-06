"""Exact dynamic-programming solver and puzzle difficulty analysis."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from math import log2


@dataclass(frozen=True)
class Solution:
    expression: str
    value: Fraction
    solution_count: int
    requires_fraction: bool
    operators: frozenset[str]


@dataclass(frozen=True)
class _Candidate:
    expression: str
    count: int
    requires_fraction: bool
    operators: frozenset[str]


def _partitions(mask: int) -> Iterable[tuple[int, int]]:
    """Yield each unordered, non-empty bipartition once."""
    left = (mask - 1) & mask
    while left:
        right = mask ^ left
        if right and left < right:
            yield left, right
        left = (left - 1) & mask


def _prefer(new: _Candidate, old: _Candidate) -> bool:
    return (
        new.requires_fraction,
        len(new.operators),
        len(new.expression),
        new.expression,
    ) < (
        old.requires_fraction,
        len(old.operators),
        len(old.expression),
        old.expression,
    )


@lru_cache(maxsize=100_000)
def _solve_values(numbers: tuple[int, ...], count_cap: int) -> dict[Fraction, _Candidate]:
    n = len(numbers)
    tables: dict[int, dict[Fraction, _Candidate]] = {}
    for index, number in enumerate(numbers):
        tables[1 << index] = {
            Fraction(number): _Candidate(str(number), 1, False, frozenset())
        }

    for size in range(2, n + 1):
        for mask in range(1, 1 << n):
            if mask.bit_count() != size:
                continue
            values: dict[Fraction, _Candidate] = {}
            for left_mask, right_mask in _partitions(mask):
                left_values = tables[left_mask]
                right_values = tables[right_mask]
                for left_value, left in left_values.items():
                    for right_value, right in right_values.items():
                        combinations = [
                            (left_value + right_value, "+", left, right),
                            (left_value * right_value, "*", left, right),
                            (left_value - right_value, "-", left, right),
                            (right_value - left_value, "-", right, left),
                        ]
                        if right_value:
                            combinations.append((left_value / right_value, "/", left, right))
                        if left_value:
                            combinations.append((right_value / left_value, "/", right, left))
                        for value, operator, first, second in combinations:
                            fractional = (
                                first.requires_fraction
                                or second.requires_fraction
                                or (operator == "/" and value.denominator != 1)
                            )
                            candidate = _Candidate(
                                f"({first.expression} {operator} {second.expression})",
                                min(count_cap, first.count * second.count),
                                fractional,
                                first.operators | second.operators | {operator},
                            )
                            existing = values.get(value)
                            if existing is None:
                                values[value] = candidate
                            else:
                                count = min(count_cap, existing.count + candidate.count)
                                best = candidate if _prefer(candidate, existing) else existing
                                values[value] = _Candidate(
                                    best.expression,
                                    count,
                                    best.requires_fraction,
                                    best.operators,
                                )
            tables[mask] = values
    return tables[(1 << n) - 1]


def solve(nums: list[int] | tuple[int, ...], target: int, count_cap: int = 10_000) -> Solution | None:
    numbers = tuple(sorted(map(int, nums)))
    candidate = _solve_values(numbers, count_cap).get(Fraction(int(target)))
    if candidate is None:
        return None
    expression = candidate.expression[1:-1] if candidate.expression.startswith("(") else candidate.expression
    return Solution(
        expression=expression,
        value=Fraction(target),
        solution_count=candidate.count,
        requires_fraction=candidate.requires_fraction,
        operators=candidate.operators,
    )


def difficulty_features(nums: list[int], target: int) -> dict[str, object]:
    solution = solve(nums, target)
    if solution is None:
        return {"solvable": False, "difficulty_score": None, "difficulty_band": "impossible"}
    rarity = max(0.0, 10.0 - log2(solution.solution_count + 1))
    fraction_penalty = 2.0 if solution.requires_fraction else 0.0
    operator_penalty = max(0, len(solution.operators) - 1) * 0.75
    score = round(rarity + fraction_penalty + operator_penalty + (len(nums) - 3), 3)
    band = "easy" if score < 5 else "medium" if score < 8 else "hard"
    return {
        "solvable": True,
        "reference_expression": solution.expression,
        "solution_count_capped": solution.solution_count,
        "requires_fraction": solution.requires_fraction,
        "operators": sorted(solution.operators),
        "difficulty_score": score,
        "difficulty_band": band,
    }

