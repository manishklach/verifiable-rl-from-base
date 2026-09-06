# Architecture and design contracts

## End-to-end flow

1. `data.py` loads the source train split, normalizes numbers and constructs plain-text prompts.
2. SHA-256 of `seed:sorted-number-multiset` assigns an entire combination to a bucket.
   Buckets 0..89 train; 90..99 evaluate. Different targets for one combination stay together.
3. Evaluation deduplicates the `(sorted nums, target)` identity before selecting a sample limit.
4. `train.py` supplies raw base-model prompts to TRL GRPO. There is no SFT stage.
5. Four independent reward functions share the strict verifier. LoRA is enabled by default.
6. `evaluate.py` samples completions, verifies each, and writes versioned JSON records.
7. `metrics.py` aggregates correctness and failure slices; `report.py` renders an HTML summary.

## Verifier boundary

Exactly one balanced, nonempty `<answer>...</answer>` block is accepted. Surrounding reasoning
is allowed; extra opening or closing answer tags invalidate the completion. Expressions are
limited to 200 characters and evaluated through a small AST allowlist using `Fraction`.
Integer literals, parentheses and binary `+ - * /` are supported. Unary signs, exponentiation,
floor division, floating-point literals, calls, names and attributes are rejected.
Negative intermediate results and fractional intermediate results are allowed. Division by
zero is invalid. Input multiplicity matters: two occurrences of 5 require two literal leaves.

Rewards are additive: format 0.10, parseability 0.25, exact number usage 0.50, correctness 2.00.
A fully correct answer earns 2.85 under `shaped`; `binary` awards 2.00 or zero. Partial rewards
can favor well-formed wrong answers. This is an intentional ablation target, not evidence of
correct reasoning. Reward metadata lengths must match; misalignment fails rather than truncates.

## Exact solver

Subset dynamic programming combines reachable rational values and stores a representative
expression per value. Each unordered partition is visited once; subtraction and division are
considered in both directions. A bounded 16-entry LRU retains completed value tables.
Public inputs are restricted to one through five nonnegative integers because complexity grows
rapidly with arity. Five-number generation can still be expensive.

The capped count is a count of enumerated derivations, not mathematically unique expressions.
Repeated input positions and equivalent derivations affect it. Operator diversity describes the
selected representative, not a guaranteed global optimum. Difficulty bands are deterministic
heuristics, not calibrated human or model difficulty. Fraction necessity is tracked by preferring
nonfractional representatives when available.

## Runtime separation

The default install provides data, verification, generation, metrics and reports. `train` adds
PyTorch, Transformers, PEFT and TRL; `plot` adds Matplotlib; `demo` adds Gradio. Install both
`train,demo` for inference UI. Model evaluation imports its GPU dependencies only when needed.
The training extra targets TRL 0.29.x. Dependency ranges are not a reproducibility lock: archive
`pip freeze`, hardware, configuration, dataset fingerprints and model revision for each real run.

## Curriculum caveat

Stages resume checkpoints with cumulative step targets 150, 300 and 500. Data skipping is
turned off when switching difficulty bands so the old stage's consumed batches do not skip
new data. Changing the total step budget also changes scheduler behavior on resume; this is
not equivalent to a single preplanned curriculum schedule. Validate stage transitions on a GPU
before using curriculum results in a comparison.
