# Evaluation protocol and output schema

## Comparable experiments

Compare base and trained models using identical held-out puzzle identities, prompt template,
seed, number of samples per prompt, temperature, token budget and verifier revision. Training's
validation split is used repeatedly for monitoring; it is not an untouched final test set. Use a
separate benchmark for final claims and audit its input combinations against training data.
Generated five-number puzzles are an arity shift from the three/four-number source; arbitrary
JSONL input does not automatically establish disjointness.

`--samples-per-prompt 1` uses greedy decoding. Larger values use stochastic decoding. A greedy
baseline and sampled candidate are not directly comparable. Use the same K for both.

## Metrics

- `strict_accuracy`: fraction of individual completions satisfying the entire verifier contract.
- `pass_at_k`: mean of `1 - C(n-c,k)/C(n,k)` across problems, where n is sample count and c is
  correct count. This order-independent estimator requires every problem to have at least k
  samples. Summary emits k=1 and the minimum available per-problem sample count.
- `parseable_rate`: permitted expressions that evaluate without error.
- `exact_number_usage_rate`: valid expressions using the full input multiset.
- `format_rate`: exactly one balanced nonempty answer block; arithmetic may still be invalid.
- `mean_completion_tokens`: generated tokens through first EOS, including EOS, excluding later padding.
- `failure_taxonomy`: correct, missing answer tag, invalid expression, wrong number usage, wrong value.
  The missing-tag bucket includes malformed/multiple blocks that fail the format contract.

Confidence intervals and multi-seed aggregation are not implemented. Do not interpret tiny
accuracy differences as statistically established improvements. Version 0.1 used prefix-hit
pass@k; those figures are not interchangeable with the version 0.2 estimator.

## JSONL benchmark input

One object per line; only `nums` and `target` are required:

```json
{"nums": [8, 3, 4], "target": 20}
```

Numbers must be one through five nonnegative integers; target must be an integer. Evaluation
rebuilds prompts and content-based IDs, deduplicates permutations with the same target, and
recomputes difficulty instead of trusting supplied annotations. Blank lines are ignored;
invalid input fails before model loading. JSONL is read into memory, so keep files manageable.

```bash
countdown-eval --model PATH_TO_MODEL --jsonl puzzles.jsonl --samples 100 \
  --samples-per-prompt 4 --seed 42 --output outputs/evaluation.json
```

## Output schema version 2

Top-level keys: `schema_version`, `model`, `seed`, `decoding`, `dataset`, `metrics`, `records`.
Decoding stores sampling mode, temperature (null for greedy), sample count and token budget.
Dataset metadata stores source and dataset fingerprint. Each completion record includes stable
problem ID, sample index, inputs, target, text, token count, expression, exact rational value
as a string, verifier flags, error and solver difficulty features.

Model paths are not immutable model revisions. Record the actual source revision externally
when running an experiment. Reports display supplied metrics and cannot certify provenance.
Archive raw JSON alongside any published chart or HTML report.
