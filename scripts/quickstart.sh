#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/setup.sh cuda
source .venv/bin/activate
countdown-doctor --config configs/qwen35-0.8b-grpo.yaml
countdown-smoke --device cuda --output "outputs/acceptance-$(date +%Y%m%d-%H%M%S)"
bash scripts/run_experiment.sh
