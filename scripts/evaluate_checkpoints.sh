#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:-outputs/qwen35-0.8b-countdown}"
SAMPLES="${SAMPLES:-500}"
K="${K:-4}"
RESULTS=()

for checkpoint in "$RUN_DIR"/checkpoint-* "$RUN_DIR"/final; do
  [[ -d "$checkpoint" ]] || continue
  name="$(basename "$checkpoint")"
  result="$RUN_DIR/eval-$name.json"
  python -m countdown_rl.evaluate \
    --model "$checkpoint" \
    --samples "$SAMPLES" \
    --samples-per-prompt "$K" \
    --output "$result"
  RESULTS+=("$result")
done

if [[ ${#RESULTS[@]} -eq 0 ]]; then
  echo "No checkpoints found beneath $RUN_DIR" >&2
  exit 1
fi

python -m countdown_rl.report "${RESULTS[@]}" \
  --output "$RUN_DIR/report.html"

