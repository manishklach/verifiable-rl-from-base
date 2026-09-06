#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
MODE="${1:-cuda}"
if [[ "$MODE" != cuda && "$MODE" != cpu ]]; then
  echo "Usage: bash scripts/setup.sh [cuda|cpu]" >&2; exit 2
fi
PYTHON="${PYTHON:-python3}"
"$PYTHON" -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,12), "Use Python 3.10-3.12"'
if [[ "$MODE" == cuda ]]; then
  command -v nvidia-smi >/dev/null || { echo "NVIDIA driver not found. Install driver first." >&2; exit 1; }
  nvidia-smi
fi
"$PYTHON" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
INDEX="https://download.pytorch.org/whl/cu128"
TORCH_VERSION="2.9.1+cu128"
if [[ "$MODE" == cpu ]]; then
  INDEX="https://download.pytorch.org/whl/cpu"
  TORCH_VERSION="2.9.1+cpu"
fi
.venv/bin/python -m pip install "torch==$TORCH_VERSION" --index-url "$INDEX"
.venv/bin/python -m pip install -c requirements/runtime.txt -e '.[train,plot,dev]'
.venv/bin/python -m pip check
.venv/bin/python -c 'import torch, transformers, trl, peft; print("torch",torch.__version__,"CUDA",torch.cuda.is_available())'
if [[ "$MODE" == cuda ]]; then
  .venv/bin/python -c 'import torch; assert torch.cuda.is_available(), "PyTorch cannot access CUDA; check driver/container passthrough"'
fi
echo "Setup complete. Run: source .venv/bin/activate"
if [[ "$MODE" == cpu ]]; then
  echo "Then: countdown-smoke --device cpu --tiny"
else
  echo "Then: countdown-smoke --device cuda"
fi
