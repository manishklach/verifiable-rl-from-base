---
base_model: Qwen/Qwen3.5-0.8B-Base
library_name: peft
pipeline_tag: text-generation
tags:
  - grpo
  - reinforcement-learning
  - verifiable-rewards
  - countdown
license: apache-2.0
---

# Qwen3.5-0.8B Countdown GRPO

LoRA adapter trained directly from `Qwen/Qwen3.5-0.8B-Base` with GRPO and deterministic
Countdown rewards. No supervised fine-tuning stage was used.

## Results

Results must be copied from the generated evaluation JSON and HTML report after the run.
Do not replace these markers with unverified estimates.

| Evaluation | Strict accuracy | pass@4 | Exact number usage |
|---|---:|---:|---:|
| Combination-disjoint test | TODO | TODO | TODO |
| Five-number OOD | TODO | TODO | TODO |

## Reward

The verifier parses a restricted arithmetic AST, checks exact multiset use of the supplied
numbers, and evaluates with rational arithmetic. It never executes model-generated code.

## Limitations

This is a task-specific research adapter. Improved Countdown accuracy is not evidence of
general reasoning, mathematical reliability, or safe deployment behavior.

