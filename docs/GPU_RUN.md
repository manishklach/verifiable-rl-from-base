# Reproducing the GPU experiment

First follow [Clone to training](GETTING_STARTED.md) and pass `countdown-smoke --device cuda`.
CUDA has not been validated on maintainer hardware.

## Recommended hardware

Start with one NVIDIA GPU with at least 24 GB VRAM. The default configuration uses LoRA,
eight generations per prompt, BF16, and gradient checkpointing. Generation—not model
weights—is normally the dominant memory and runtime cost.

For a smaller GPU, reduce these values in the YAML configuration in this order:

1. `max_completion_length`: 256 → 128
2. `per_device_train_batch_size`: 2 → 1
3. `num_generations`: 8 → 4

Keep the effective batch large by increasing `gradient_accumulation_steps`. The effective
number of generated sequences per optimizer step is approximately:

```text
devices × batch size × gradient accumulation × generations
```

This is the rollout sequence batch, already including repeated generations. Divide by
`num_generations` for distinct prompts. It must be divisible by `num_generations`.
Evaluation sets its per-device batch to `num_generations`. `WORLD_SIZE` controls the
preflight process count; visible GPUs alone do not imply distributed execution.

## RunPod or another GPU VM

```bash
git clone https://github.com/manishklach/verifiable-rl-from-base.git
cd verifiable-rl-from-base
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
bash scripts/setup.sh cuda
source .venv/bin/activate
countdown-smoke --device cuda
bash scripts/run_experiment.sh
```

Set `report_to: wandb` in the experiment YAML and run `wandb login` when remote tracking
is desired. Never commit the API key.

## Docker

```bash
docker build -t countdown-grpo .
docker run --rm --gpus all --ipc=host --shm-size=16g \
  -v "$PWD/outputs:/workspace/verifiable-rl-from-base/outputs" \
  countdown-grpo
```

The container defaults to acceptance. To run the full experiment, append
`bash scripts/run_experiment.sh` to the docker run command. Docker is not locally validated.

## After training

```bash
SAMPLES=500 K=4 bash scripts/evaluate_checkpoints.sh
python -m countdown_rl.generate --arity 5 --samples 500
python -m countdown_rl.evaluate \
  --model outputs/qwen35-0.8b-countdown/final \
  --jsonl data/ood-five-number.jsonl \
  --samples 500 --samples-per-prompt 4 \
  --output outputs/qwen35-0.8b-countdown/eval-ood-five.json
```

Do not publish the headline result until the base model and trained checkpoint have been
evaluated with identical decoding settings across at least three seeds.

