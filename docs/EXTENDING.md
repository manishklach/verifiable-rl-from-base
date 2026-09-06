# Build on top of the project

This guide maps common ideas to the smallest extension point that implements them. Start
with the narrowest surface possible; the existing pipeline will supply splitting,
verification, metrics, and reporting around it.

## Mental model

```text
Hugging Face rows
    → grouped train/eval split
    → prompt construction
    → model samples N completions
    → independent reward functions
    → GRPO policy update
    → strict held-out verification
    → metrics and HTML report
```

The training oracle and evaluation oracle share the same arithmetic verifier. That makes
consistency easy to audit, but it also means verifier changes require exceptional care.

## Repository map

| You want to change… | Start here | Contract to preserve |
|---|---|---|
| Prompt wording | `src/countdown_rl/prompts.py` | End with one parseable answer block |
| Correctness rules | `src/countdown_rl/verifier.py` | Never execute generated code |
| Reward shaping | `src/countdown_rl/rewards.py` | Return one float per completion |
| Dataset or split | `src/countdown_rl/data.py` | Keep related combinations disjoint |
| Puzzle difficulty | `src/countdown_rl/solver.py` | Use exact `Fraction` arithmetic |
| Training parameters | `configs/` | Preserve config provenance |
| Evaluation metrics | `src/countdown_rl/metrics.py` | Aggregate by stable problem ID |
| OOD benchmarks | `src/countdown_rl/generate.py` | Generate deterministically |
| Report presentation | `src/countdown_rl/report.py` | Escape all model-generated text |
| Interactive UI | `src/countdown_rl/demo.py` | Show the verifier verdict |

## Add a reward function

Create a function in `rewards.py`:

```python
def concise_correct_reward(completions, nums, target, **kwargs) -> list[float]:
    rewards = []
    for text, result in _rows(completions, nums, target):
        rewards.append(0.2 if result.correct and len(text) < 160 else 0.0)
    return rewards
```

Then add a named profile in `reward_functions`, expose it through a YAML configuration,
and add tests proving that an incorrect completion cannot earn the correctness-gated
bonus.

Avoid rewarding short output without gating on correctness; otherwise the policy can earn
the bonus by returning an empty or useless completion.

## Add a model

Copy the main configuration:

```bash
cp configs/qwen35-0.8b-grpo.yaml configs/my-model.yaml
```

Change `model_name` and `output_dir`, then run:

```bash
countdown-doctor --config configs/my-model.yaml
accelerate launch -m countdown_rl.train --config configs/my-model.yaml
```

The evaluator recognizes ordinary causal-LM checkpoints and Qwen-style
`ForConditionalGeneration` wrappers. A genuinely new architecture may require a loader
branch and an integration test.

## Add a dataset

Normalize every row to at least:

```python
{
    "nums": [8, 3, 4],
    "target": 20,
    "prompt": "...",
}
```

Define a stable problem identity and group all equivalent or near-duplicate variants into
the same split. Add an assertion showing that grouped train and evaluation identities do
not overlap.

## Add an evaluation metric

Metrics operate on flat completion records. Each record includes `problem_id`,
`sample_index`, verifier flags, completion length, arity, and difficulty metadata.

Add the aggregation to `summarize` and a two-problem unit test. For sampling metrics,
aggregate within `problem_id` before averaging across problems; treating every completion
as an independent problem produces misleading `pass@k` results.

## Add an OOD suite

Use a fixed seed, store a stable problem ID, and prove each generated task is solvable with
the exact solver. Keep generation independent of model output. Evaluate it with:

```bash
countdown-eval \
  --model outputs/qwen35-0.8b-countdown/final \
  --jsonl data/your-suite.jsonl \
  --samples-per-prompt 4 \
  --output outputs/eval-your-suite.json
```

## Run a cheap development cycle

Most changes do not require a GPU:

```bash
ruff check .
pytest -q
python -m countdown_rl.generate \
  --samples 20 --arity 5 --minimum 1 --maximum 20 \
  --output /tmp/countdown-smoke.jsonl
```

For training changes, use a temporary YAML with `max_steps: 2`, a small dataset, and short
completions. A smoke test establishes that the pipeline runs; it is not an experimental
result.

## Compare an experiment fairly

Use the same:

- held-out problem IDs;
- model revision;
- sample count and `samples-per-prompt`;
- temperature and maximum completion length;
- seed set;
- verifier revision.

Then generate a comparison report:

```bash
countdown-report \
  outputs/baseline.json outputs/candidate.json \
  --title "Baseline versus candidate" \
  --output reports/comparison.html
```

## Definition of done

A contribution is ready when:

- its behavior is documented;
- normal and adversarial paths are tested;
- local checks pass;
- generated artifacts and secrets are absent from the diff;
- experimental claims have reproducible evidence;
- the pull-request template is complete.

