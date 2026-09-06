"""Deterministic generation of out-of-distribution Countdown benchmarks."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .prompts import make_prompt
from .solver import difficulty_features


def generate_benchmark(
    samples: int,
    arity: int,
    minimum: int,
    maximum: int,
    seed: int,
) -> list[dict[str, object]]:
    if samples < 1 or not 1 <= arity <= 5 or not 1 <= minimum <= maximum:
        raise ValueError("require samples > 0, arity 1..5, and 1 <= minimum <= maximum")
    rng = random.Random(seed)
    rows: list[dict[str, object]] = []
    seen: set[tuple[tuple[int, ...], int]] = set()
    attempts = 0
    while len(rows) < samples and attempts < samples * 100:
        attempts += 1
        nums = [rng.randint(minimum, maximum) for _ in range(arity)]
        # Draw a reachable target by first analyzing a random target range. This avoids
        # leaking a construction template while ensuring the benchmark is solvable.
        target = rng.randint(1, maximum * 2)
        key = (tuple(sorted(nums)), target)
        if key in seen:
            continue
        features = difficulty_features(nums, target)
        if not features["solvable"]:
            continue
        seen.add(key)
        rows.append(
            {
                "problem_id": f"ood-{seed}-{len(rows):05d}",
                "nums": nums,
                "target": target,
                "prompt": make_prompt(nums, target),
                **features,
            }
        )
    if len(rows) != samples:
        raise RuntimeError(f"generated only {len(rows)} of {samples} requested puzzles")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--arity", type=int, default=5)
    parser.add_argument("--minimum", type=int, default=1)
    parser.add_argument("--maximum", type=int, default=100)
    parser.add_argument("--seed", type=int, default=31415)
    parser.add_argument("--output", default="data/ood-five-number.jsonl")
    args = parser.parse_args()
    rows = generate_benchmark(args.samples, args.arity, args.minimum, args.maximum, args.seed)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("".join(json.dumps(row) + "\n" for row in rows))
    print(f"Wrote {len(rows)} puzzles to {destination}")


if __name__ == "__main__":
    main()
