# Contributing to Verifiable RL from Base

Thank you for improving the project. Contributions are welcome across reward design,
training infrastructure, evaluation, documentation, visualizations, and reproducibility.

This repository treats experimental integrity as part of correctness. A change is not
complete merely because it runs: it must preserve strict verification, prevent train/test
leakage, and distinguish measured results from hypotheses.

## The shortest contribution path

1. Fork the repository on GitHub.
2. Clone your fork and create a focused branch.
3. Install the development environment.
4. Make the change and add tests.
5. Run the local quality checks.
6. Commit, push, and open a pull request.

```bash
git clone https://github.com/YOUR_USERNAME/verifiable-rl-from-base.git
cd verifiable-rl-from-base
git remote add upstream https://github.com/manishklach/verifiable-rl-from-base.git

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"

git switch -c feature/short-description
```

Before committing:

```bash
ruff check .
pytest -q
python -m compileall -q src tests
git diff --check
```

Then contribute the branch back:

```bash
git add path/to/the/files-you-changed
git commit -m "Add concise description of change"
git push -u origin feature/short-description
```

Open a pull request from your fork into `manishklach/verifiable-rl-from-base:main`.
If you use GitHub CLI, the final step is:

```bash
gh pr create --fill --repo manishklach/verifiable-rl-from-base
```

## Keep your fork current

Do this before starting a new contribution:

```bash
git fetch upstream
git switch main
git merge --ff-only upstream/main
git push origin main
```

If your feature branch is already open, merge the updated `main` into it and resolve only
the conflicts related to your work.

## What makes a good pull request

A strong pull request:

- solves one clearly stated problem;
- explains why the change is needed;
- includes tests for new behavior and failure cases;
- contains no generated model weights, caches, credentials, or large result dumps;
- records the exact configuration for any claimed experiment;
- compares models under identical splits and decoding settings;
- does not claim “reasoning” from a few appealing examples.

Small pull requests are easier to review and reproduce. Refactors and behavioral changes
should normally be separate pull requests.

## Experimental result requirements

When a pull request reports training results, include:

1. Base model and exact revision, when pinned.
2. Dataset name, fingerprint, split seed, and row counts.
3. Hardware and relevant package versions.
4. Full YAML configuration.
5. Baseline and trained-model results under identical decoding.
6. Number of independent seeds.
7. Strict accuracy, valid-expression rate, exact-number-use rate, and `pass@k`.
8. At least a brief failure analysis.

Result JSON and the generated HTML report may be attached to the pull request. Do not
commit very large completion logs to Git.

## Changes requiring particular care

### Verifier changes

The verifier is a security boundary. Never execute generated expressions with `eval`,
`exec`, a shell, or a Python interpreter. Extend the AST allowlist explicitly and add both
positive and adversarial tests.

### Reward changes

Every new reward must document:

- exactly what behavior it rewards;
- its numeric range;
- known ways a model could exploit it;
- whether it can grant reward to an incorrect answer.

Reward functions must accept `**kwargs` so they remain compatible with TRL metadata.

### Dataset changes

Do not introduce row-level random splitting. Group related examples by the sorted input
multiset or a stricter task-specific identity. Add a leakage assertion whenever a new split
strategy is introduced.

### GPU training changes

Run `countdown-doctor` first. Smoke-test with a tiny number of steps before launching the
full experiment. Never commit Hugging Face or Weights & Biases tokens.

## Commit style

Use an imperative subject describing the outcome:

```text
Add pass@k confidence intervals
Harden verifier against exponentiation
Document multi-GPU evaluation
```

Avoid commits such as `changes`, `updates`, or `fix stuff`. A pull request may contain
several logical commits, but each should leave the repository understandable.

## Getting help

- Use a bug report for reproducible failures.
- Use an experiment proposal for a new reward, curriculum, dataset, or ablation.
- Open a draft pull request when early architectural feedback would prevent wasted work.

See [`docs/EXTENDING.md`](docs/EXTENDING.md) for concrete code examples and
[`docs/GPU_RUN.md`](docs/GPU_RUN.md) for the training workflow.

