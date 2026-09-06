#!/usr/bin/env bash
set -euo pipefail

MODEL="Qwen/Qwen3.5-0.8B-Base"
RUN_DIR="outputs/qwen35-0.8b-countdown"

python -m countdown_rl.doctor --config configs/qwen35-0.8b-grpo.yaml

python -m countdown_rl.evaluate \
  --model "$MODEL" \
  --samples 500 \
  --output "$RUN_DIR/baseline.json"

accelerate launch -m countdown_rl.train \
  --config configs/qwen35-0.8b-grpo.yaml

python -m countdown_rl.evaluate \
  --model "$RUN_DIR/final" \
  --samples 500 \
  --output "$RUN_DIR/final-eval.json"

python -m countdown_rl.plot \
  --state "$RUN_DIR/trainer_state.json" \
  --output assets/training-curves.png

python -m countdown_rl.report \
  "$RUN_DIR/baseline.json" "$RUN_DIR/final-eval.json" \
  --output "$RUN_DIR/report.html"
