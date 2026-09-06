"""Composable rewards for TRL's GRPOTrainer."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .verifier import answer_block, verify_completion


def _text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        item = completion[-1]
        if isinstance(item, dict):
            return str(item.get("content", ""))
    return str(completion)


def _rows(completions: Sequence[Any], nums: Sequence[Sequence[int]], target: Sequence[int]):
    for completion, numbers, goal in zip(completions, nums, target, strict=True):
        text = _text(completion)
        yield text, verify_completion(text, numbers, int(goal))


def format_reward(completions, **kwargs) -> list[float]:
    """Small incentive for emitting exactly one non-empty answer block."""
    rewards = []
    for completion in completions:
        rewards.append(0.1 if answer_block(_text(completion)) is not None else 0.0)
    return rewards


def parseable_reward(completions, nums, target, **kwargs) -> list[float]:
    return [0.25 if result.parseable else 0.0 for _, result in _rows(completions, nums, target)]


def number_usage_reward(completions, nums, target, **kwargs) -> list[float]:
    return [
        0.5 if result.uses_numbers_exactly_once else 0.0
        for _, result in _rows(completions, nums, target)
    ]


def correctness_reward(completions, nums, target, **kwargs) -> list[float]:
    return [2.0 if result.correct else 0.0 for _, result in _rows(completions, nums, target)]


REWARD_FUNCTIONS = [format_reward, parseable_reward, number_usage_reward, correctness_reward]


def reward_functions(profile: str):
    if profile == "shaped":
        return REWARD_FUNCTIONS
    if profile == "binary":
        return [correctness_reward]
    if profile == "no-format":
        return [parseable_reward, number_usage_reward, correctness_reward]
    raise ValueError(f"unknown reward profile: {profile}")
