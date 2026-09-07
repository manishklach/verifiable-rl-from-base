# Verifiable RL from Base

**Can rule-based reinforcement learning elicit arithmetic search from a 0.8B base
model—without a supervised fine-tuning warm start?**

This repository trains `Qwen/Qwen3.5-0.8B-Base` directly with Group Relative Policy
Optimization (GRPO) on Jiayi Pan's Countdown tasks. There is no SFT stage and no learned
reward model. Every reward is computed by a strict symbolic verifier.

It is an experiment platform rather than a single training file: exact puzzle solving,
difficulty-aware curricula, adversarial reward audits, `pass@k`, out-of-distribution
benchmarks, checkpoint comparisons, an interactive demo, and a self-contained research
report are included.

> Status: v0.3.0 adds real trainer integration and a hardware acceptance command.
> GPU performance and model accuracy improvements are not yet demonstrated.

[![CI](https://github.com/manishklach/verifiable-rl-from-base/actions/workflows/ci.yml/badge.svg)](https://github.com/manishklach/verifiable-rl-from-base/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Run on real hardware

For Linux with one NVIDIA GPU and Python 3.10-3.12:

```bash
git clone https://github.com/manishklach/verifiable-rl-from-base.git
cd verifiable-rl-from-base
bash scripts/setup.sh cuda
source .venv/bin/activate
countdown-smoke --device cuda
bash scripts/run_experiment.sh
```

Or run all stages with `bash scripts/quickstart.sh`. Read the
[clone-to-training guide](docs/GETTING_STARTED.md) for prerequisites and validation scope.
Tiny Qwen hybrid training passes CPU integration; CUDA still needs an actual GPU test.

## Try it without a GPU

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv/Scripts/Activate.ps1
pip install -e ".[dev]"
pytest -q
countdown-generate --samples 3 --arity 3 --maximum 10 --output outputs/smoke.jsonl
```

```python
from countdown_rl.solver import solve
from countdown_rl.verifier import verify_completion
solution = solve([8, 3, 4], 20)
print(solution.expression)
assert verify_completion(f"<answer>{solution.expression}</answer>", [8, 3, 4], 20).correct
```

The solver demonstration verifies software behavior; it is not a language-model result.

## Documentation

| Guide | What it covers |
|---|---|
| [Architecture](docs/ARCHITECTURE.md) | Data flow, verifier, rewards, solver limits and curriculum behavior |
| [Evaluation](docs/EVALUATION.md) | pass@k definition, fair comparisons and JSON/JSONL schemas |
| [GPU run](docs/GPU_RUN.md) | Installation, hardware, Docker and experiment commands |
| [Troubleshooting](docs/TROUBLESHOOTING.md) | CPU Windows setup, common failures and GPU acceptance |
| [Extension guide](docs/EXTENDING.md) | Add datasets, models, rewards and metrics |
| [Contributing](CONTRIBUTING.md) | Development workflow and evidence requirements |
| [Changelog](CHANGELOG.md) | Release changes and migration notes |

## Start here

- **Run the project:** follow the [GPU experiment guide](docs/GPU_RUN.md).
- **Extend rewards, models, datasets, or metrics:** read [Build on top](docs/EXTENDING.md).
- **Contribute changes upstream:** follow [CONTRIBUTING.md](CONTRIBUTING.md).

## Why this experiment

Countdown provides a clean test bed for verifiable reinforcement learning:

- exploration is performed by the language model;
- correctness can be checked deterministically;
- reward hacking can be measured and blocked;
- progress can be compared against the untouched base model.

The claim under test is **not** that the model spontaneously acquires general reasoning.
The narrower hypothesis is that outcome-based RL can increase valid arithmetic search
behavior in a small pretrained base model without demonstrations.

## Strict reward contract

A completion is correct only when its `<answer>` expression:

1. parses as an arithmetic syntax tree;
2. contains only integer literals, `+`, `-`, `*`, `/`, and parentheses;
3. uses every supplied number exactly once, including duplicates;
4. introduces no other numbers; and
5. evaluates exactly to the target using rational arithmetic.

Python execution and `eval` are never used. Merely printing the target earns no
correctness reward.

| Component | Reward |
|---|---:|
| One non-empty `<answer>` block | 0.10 |
| Parseable, permitted expression | 0.25 |
| Exact multiset of supplied numbers | 0.50 |
| Exact target value | 2.00 |

## Leakage-resistant evaluation

The source dataset contains related examples and permutations. A naive row-level random
split can therefore inflate accuracy. This project hashes the **sorted multiset of input
numbers** and assigns the whole group to train or evaluation. A number combination cannot
cross the boundary, even if its order differs.

## Exact solver and meaningful difficulty

The dynamic-programming solver enumerates reachable rational values over subsets of the
input multiset. It proves solvability and emits a verifier-compatible reference expression.
Difficulty is based on solution rarity, operator diversity, and whether fractional
intermediate values are required—not merely expression depth, which is almost constant
when every number must be used.

Three staged curriculum configurations are supplied:

```bash
bash scripts/run_curriculum.sh  # easy → medium → hard, resuming optimizer state
```

## Quick start

Use Linux with a recent NVIDIA GPU. A 24 GB GPU is a practical starting point for the
default LoRA configuration; reduce completion length or batch size if needed.

```bash
git clone https://github.com/manishklach/verifiable-rl-from-base.git
cd verifiable-rl-from-base
python -m venv .venv
source .venv/bin/activate
pip install -c requirements/runtime.txt -e ".[train,plot,dev,wandb]"
```

Evaluate the untouched base model first:

```bash
countdown-eval \
  --model Qwen/Qwen3.5-0.8B-Base \
  --samples 500 \
  --output outputs/baseline.json
```

Train for 500 GRPO steps:

```bash
python -m countdown_rl.train \
  --config configs/qwen35-0.8b-grpo.yaml
```

Or execute the baseline, training, final evaluation, and plot pipeline together:

```bash
bash scripts/run_experiment.sh
```

The pipeline begins with `countdown-doctor`, which checks CUDA, BF16 support, dependency
versions, Qwen's actual model architecture, and GRPO batch divisibility before downloading
weights or committing GPU hours.

Evaluate the adapter and plot reward curves:

```bash
countdown-eval \
  --model outputs/qwen35-0.8b-countdown/final \
  --samples 500 --samples-per-prompt 4 \
  --output outputs/final.json

countdown-plot \
  --state outputs/qwen35-0.8b-countdown/trainer_state.json \
  --output assets/training-curves.png
```

Evaluate every saved checkpoint and produce a standalone HTML report:

```bash
SAMPLES=500 K=4 bash scripts/evaluate_checkpoints.sh
```

Generate a solvable five-number OOD benchmark and evaluate it:

```bash
countdown-generate --arity 5 --samples 500
countdown-eval \
  --model outputs/qwen35-0.8b-countdown/final \
  --jsonl data/ood-five-number.jsonl \
  --samples 500 --samples-per-prompt 4 \
  --output outputs/eval-ood-five.json
```

Launch the verifier-backed side-by-side demo after training:

```bash
pip install -e ".[train,demo]"
countdown-demo --trained outputs/qwen35-0.8b-countdown/final
```

## Experimental protocol

Run at least three seeds before making a strong claim. Record:

| Measurement | Base | Step 50 | Step 100 | Step 250 | Step 500 |
|---|---:|---:|---:|---:|---:|
| Strict held-out accuracy | TBD | TBD | TBD | TBD | TBD |
| Valid-expression rate | TBD | TBD | TBD | TBD | TBD |
| Exact-number-usage rate | TBD | TBD | TBD | TBD | TBD |
| Mean completion tokens | TBD | TBD | TBD | TBD | TBD |

Recommended follow-up ablations:

- binary correctness reward versus the shaped reward;
- LoRA versus full-parameter updates;
- three-number versus four-number tasks;
- group split versus naive row split;
- zero-SFT versus a small cold-start SFT set.

Ready-to-run configurations for binary reward, removal of the format reward, and stronger
KL regularization live under `configs/ablations/`. Each run must use the same held-out
problem IDs and decoding configuration for a valid comparison.

## Repository layout

```text
configs/                       GRPO experiment configuration
src/countdown_rl/data.py       grouped dataset split and prompts
src/countdown_rl/solver.py     exact DP solver and difficulty analysis
src/countdown_rl/verifier.py   safe AST evaluator and strict checker
src/countdown_rl/rewards.py    composable GRPO rewards
src/countdown_rl/train.py      zero-SFT training entry point
src/countdown_rl/evaluate.py   pass@k, OOD, slices, and failure analysis
src/countdown_rl/generate.py   deterministic solvable OOD generation
src/countdown_rl/report.py     self-contained HTML experiment report
src/countdown_rl/demo.py       interactive base-versus-GRPO comparison
src/countdown_rl/plot.py       training-curve generator
tests/                         unit, adversarial, and property tests
```

See [`docs/GPU_RUN.md`](docs/GPU_RUN.md) for RunPod/Docker instructions and memory knobs.

## Reproducibility notes

- The raw Hugging Face dataset is downloaded at run time.
- The exact experiment configuration is copied to the output directory.
- Dataset fingerprints, row counts, split seed, and split rule are recorded per run.
- Checkpoints are written every 50 steps.
- The final adapter, tokenizer, and trainer state are saved together.
- Evaluation uses greedy decoding by default for repeatability.

## License

MIT

## Installation extras

| Extra | Purpose |
|---|---|
| `dev` | pytest, Hypothesis and Ruff; no GPU packages |
| `train` | GRPO training and model evaluation |
| `plot` | Training-curve charts |
| `demo` | Gradio UI; also install `train` |
| `wandb` | Optional experiment tracking |

## Known limits and research claims

Difficulty is heuristic; solution counts are capped derivation counts. The solver accepts at
most five numbers and can be slow on five-number tasks. Held-out monitoring data is not an
untouched final test set. The HTML report does not verify that runs are comparable. Model and
dataset dependencies are fetched at runtime; archive exact versions for reproducibility.
No trained checkpoint or headline accuracy result is included. The 24 GB suggestion is a
starting configuration, not a measured memory guarantee.

## References

- [TRL GRPO documentation](https://huggingface.co/docs/trl/v0.29.0/en/grpo_trainer)
- [Countdown source dataset](https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4)
- [Configured Qwen base model](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base)
- [Software citation metadata](CITATION.cff)
