from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True)
    parser.add_argument("--output", default="assets/training-curves.png")
    args = parser.parse_args()
    history = json.loads(Path(args.state).read_text())["log_history"]
    rows = [row for row in history if "step" in row]
    reward_keys = sorted({key for row in rows for key in row if key.startswith("rewards/")})
    if not reward_keys:
        raise SystemExit("No per-reward metrics found in trainer state")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for key in reward_keys:
        points = [(row["step"], row[key]) for row in rows if key in row]
        ax.plot([p[0] for p in points], [p[1] for p in points], label=key.removeprefix("rewards/"))
    ax.set(xlabel="GRPO step", ylabel="Mean reward", title="Countdown GRPO training")
    ax.grid(alpha=0.25)
    ax.legend()
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    print(destination)


if __name__ == "__main__":
    main()
