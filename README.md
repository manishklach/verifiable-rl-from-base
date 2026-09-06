# Verifiable RL from Base

**Can rule-based reinforcement learning elicit arithmetic search from a 0.8B base
model—without a supervised fine-tuning warm start?**

This repository trains `Qwen/Qwen3.5-0.8B-Base` directly with Group Relative Policy
Optimization (GRPO) on Jiayi Pan's Countdown tasks. There is no SFT stage and no learned
reward model. Every reward is computed by a strict symbolic verifier.

> Status: implementation complete; experiment results are intentionally not claimed until
> the GPU run and held-out evaluation have finished.

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

## Quick start

Use Linux with a recent NVIDIA GPU. A 24 GB GPU is a practical starting point for the
default LoRA configuration; reduce completion length or batch size if needed.

```bash
git clone https://github.com/manishklach/verifiable-rl-from-base.git
cd verifiable-rl-from-base
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,wandb]"
accelerate config
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
accelerate launch -m countdown_rl.train \
  --config configs/qwen35-0.8b-grpo.yaml
```

Or execute the baseline, training, final evaluation, and plot pipeline together:

```bash
bash scripts/run_experiment.sh
```

Evaluate the adapter and plot reward curves:

```bash
countdown-eval \
  --model outputs/qwen35-0.8b-countdown/final \
  --samples 500 \
  --output outputs/final.json

countdown-plot \
  --state outputs/qwen35-0.8b-countdown/trainer_state.json \
  --output assets/training-curves.png
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

## Repository layout

```text
configs/                       GRPO experiment configuration
src/countdown_rl/data.py       grouped dataset split and prompts
src/countdown_rl/verifier.py   safe AST evaluator and strict checker
src/countdown_rl/rewards.py    composable GRPO rewards
src/countdown_rl/train.py      zero-SFT training entry point
src/countdown_rl/evaluate.py   deterministic held-out evaluation
src/countdown_rl/plot.py       training-curve generator
tests/                         verifier and split invariants
```

## Reproducibility notes

- The raw Hugging Face dataset is downloaded at run time.
- The exact experiment configuration is copied to the output directory.
- Checkpoints are written every 50 steps.
- The final adapter, tokenizer, and trainer state are saved together.
- Evaluation uses greedy decoding by default for repeatability.

## License

MIT
