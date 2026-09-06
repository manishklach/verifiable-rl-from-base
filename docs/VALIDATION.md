# Release validation

Version 0.2.0 was checked on Windows with Python 3.11 in a fresh virtual environment.

- 48 offline unit, adversarial, property and regression tests passed.
- Ruff checks passed; Python source and tests compile successfully.
- A deterministic three-puzzle generation smoke run completed.
- Wheel and source-distribution builds completed; pip dependency checks passed.
- TRL 0.29.0 wheel source was inspected to confirm removal of max_prompt_length.
- GitHub CI repeats tests, build and generation on Linux/Windows with Python 3.10/3.12.

Tests cover exact rational verification, duplicate operands, solver witnesses, malformed
answer blocks, forbidden syntax, reward alignment, configuration errors, deterministic OOD
generation, held-out deduplication, JSONL prompt reconstruction and pass@k order independence.

Not executed locally: model weight download, CUDA training, adapter inference, distributed
training, curriculum resume, Docker build and browser demo. Those require integration validation;
passing CPU tests does not establish model quality or full GPU runtime compatibility.
