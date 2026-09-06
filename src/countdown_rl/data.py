"""Dataset preparation with group-based splits to prevent combination leakage."""

from __future__ import annotations

import hashlib

from datasets import Dataset, load_dataset

from .prompts import make_prompt
from .solver import difficulty_features


def combination_key(nums: list[int]) -> str:
    return ",".join(map(str, sorted(map(int, nums))))


def split_bucket(nums: list[int], seed: int = 42) -> int:
    payload = f"{seed}:{combination_key(nums)}".encode()
    return int(hashlib.sha256(payload).hexdigest()[:8], 16) % 100


def assert_combination_disjoint(train: Dataset, evaluation: Dataset) -> None:
    train_keys = {combination_key(nums) for nums in train["nums"]}
    eval_keys = {combination_key(nums) for nums in evaluation["nums"]}
    overlap = train_keys & eval_keys
    if overlap:
        raise AssertionError(f"train/eval combination leakage detected: {len(overlap)} groups")


def prepare_dataset(
    dataset_name: str,
    train_size: int | None = None,
    eval_size: int | None = None,
    seed: int = 42,
    annotate_difficulty: bool = False,
) -> tuple[Dataset, Dataset]:
    for name, size in (("train_size", train_size), ("eval_size", eval_size)):
        if size is not None and (type(size) is not int or size < 1):
            raise ValueError(f"{name} must be a positive integer or None")
    raw = load_dataset(dataset_name, split="train")

    def enrich(row):
        nums = [int(value) for value in row["nums"]]
        target = int(row["target"])
        return {
            "nums": nums,
            "target": target,
            "prompt": make_prompt(nums, target),
            "_bucket": split_bucket(nums, seed),
        }

    data = raw.map(enrich, desc="Building prompts and split keys")
    train = data.filter(lambda row: row["_bucket"] < 90, desc="Selecting train groups")
    evaluation = data.filter(lambda row: row["_bucket"] >= 90, desc="Selecting eval groups")
    train = train.shuffle(seed=seed)
    evaluation = evaluation.shuffle(seed=seed + 1)
    seen = set()
    indices = []
    for index, row in enumerate(evaluation):
        key = (combination_key(row["nums"]), row["target"])
        if key not in seen:
            seen.add(key)
            indices.append(index)
    evaluation = evaluation.select(indices)
    if train_size:
        train = train.select(range(min(train_size, len(train))))
    if eval_size:
        evaluation = evaluation.select(range(min(eval_size, len(evaluation))))
    train = train.remove_columns("_bucket")
    evaluation = evaluation.remove_columns("_bucket")
    if annotate_difficulty:
        train = train.map(
            lambda row: difficulty_features(row["nums"], row["target"]),
            desc="Analyzing train difficulty",
        ).filter(lambda row: row["solvable"], desc="Removing impossible train puzzles")
        evaluation = evaluation.map(
            lambda row: difficulty_features(row["nums"], row["target"]),
            desc="Analyzing eval difficulty",
        ).filter(lambda row: row["solvable"], desc="Removing impossible eval puzzles")
    assert_combination_disjoint(train, evaluation)
    return train, evaluation
