"""Dataset preparation with group-based splits to prevent combination leakage."""

from __future__ import annotations

import hashlib

from datasets import Dataset, load_dataset

from .prompts import make_prompt


def combination_key(nums: list[int]) -> str:
    return ",".join(map(str, sorted(map(int, nums))))


def split_bucket(nums: list[int], seed: int = 42) -> int:
    payload = f"{seed}:{combination_key(nums)}".encode()
    return int(hashlib.sha256(payload).hexdigest()[:8], 16) % 100


def prepare_dataset(
    dataset_name: str,
    train_size: int | None = None,
    eval_size: int | None = None,
    seed: int = 42,
) -> tuple[Dataset, Dataset]:
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
    if train_size:
        train = train.select(range(min(train_size, len(train))))
    if eval_size:
        evaluation = evaluation.select(range(min(eval_size, len(evaluation))))
    return train.remove_columns("_bucket"), evaluation.remove_columns("_bucket")

