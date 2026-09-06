#!/usr/bin/env bash
set -euo pipefail

accelerate launch -m countdown_rl.train --config configs/curriculum/easy.yaml
accelerate launch -m countdown_rl.train --config configs/curriculum/medium.yaml
accelerate launch -m countdown_rl.train --config configs/curriculum/hard.yaml

