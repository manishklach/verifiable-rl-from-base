# Clone to first training run

## Target environment

The primary setup path is Linux, Python 3.10–3.12, and one NVIDIA GPU with a driver
compatible with CUDA 12.8. Ubuntu 22.04/24.04 is the intended host. Start with 24 GB
VRAM, 32 GB host RAM and 20 GB free disk for downloads and checkpoints. These are
planning budgets, not measured guarantees. Run hardware acceptance on your host.
Internet access to PyPI, download.pytorch.org and Hugging Face is needed initially.

No local CUDA toolkit or compiled Flash Attention extension is required. PyTorch's
wheel provides the runtime. Qwen's optional fast linear-attention extensions are
omitted; its PyTorch fallback avoids compilation, but can be slower.

## One command after cloning

```bash
git clone https://github.com/manishklach/verifiable-rl-from-base.git
cd verifiable-rl-from-base
bash scripts/quickstart.sh
```

This creates `.venv`, installs the constrained runtime and CUDA PyTorch wheel,
checks CUDA, runs the acceptance test, then starts baseline evaluation, 500-step
training, final evaluation, plotting and reporting. Each failure stops the next
stage. It does not install system drivers or rent hardware. Keep the terminal open.

## Recommended first use: run each stage separately

```bash
bash scripts/setup.sh cuda
source .venv/bin/activate
countdown-doctor --config configs/qwen35-0.8b-grpo.yaml
countdown-smoke --device cuda
# Only after acceptance.json reports passed:
bash scripts/run_experiment.sh
```

No `accelerate config` is needed for this single-GPU workflow. Distributed training
is outside this setup path. Existing checkpoints require explicit resume or a new
output directory. Smoke refuses to overwrite evidence; use `--output NEW_PATH`.

## What acceptance checks

The CUDA test downloads the real Qwen base model and source Countdown dataset,
selects tiny train/eval subsets, performs LoRA GRPO with the normal rewards, saves
checkpoint-1, resumes to step 2, reloads the adapter and evaluates two held-out
problems. It creates `report.html`, `curves.png` and `acceptance.json`, including
hardware and versions. It uses shorter completions and fewer generations than the
full run; passing does not establish full-configuration VRAM requirements. Zero
reward after two steps is possible and is not itself a runtime failure.

## CPU integration test

```bash
bash scripts/setup.sh cpu
source .venv/bin/activate
countdown-smoke --device cpu --tiny
```

The tiny test generates a local random Qwen3.5 checkpoint with both linear and full
attention. It verifies exact text-weight loading from the composite checkpoint,
nonzero LoRA updates, checkpoint resume, adapter reload and artifact generation.
An explicitly labeled diagnostic reward creates nonzero gradients for random text;
it is never used by real-model acceptance or normal training. No model/dataset
downloads are needed. Tiny-model accuracy has no scientific meaning.

Windows CPU users can install the same constraints with `.venv/Scripts/python`;
Bash scripts require Linux/WSL. CPU success does not certify CUDA kernels or VRAM.

## Runtime versions

`requirements/runtime.txt` fixes PyTorch 2.9.1, Transformers 5.3.0, TRL 0.29.0,
PEFT 0.18.1, Accelerate 1.12.0, Datasets 5.0.1, Matplotlib 3.10.8 and PyYAML 6.0.3.
PyTorch is installed first from the device-specific index. Transitive dependencies
remain resolver-selected. Archive `pip freeze` and `acceptance.json` for your exact
environment; rerun acceptance after changing dependencies.

## Validation status

Tiny hybrid-model integration passes locally on Windows CPU and is exercised by
Linux CI. No NVIDIA GPU is available for this release: real-model CUDA execution
and full-run memory/time remain unmeasured. The acceptance command produces actual
host evidence instead of treating unit tests as proof of GPU compatibility.
