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

## Version 0.3.0 integration

The tiny Qwen3.5 hybrid checkpoint now runs through the production training function on
Windows CPU with PyTorch 2.9.1+cpu, Transformers 5.3.0, TRL 0.29.0, PEFT 0.18.1,
Accelerate 1.12.0 and Datasets 5.0.1. Both linear and full attention are present.
The test checks text-weight equality, nonzero adapter updates (using an explicitly
integration-only diagnostic reward), checkpoint resume to step 2, saved-adapter inference,
two evaluation records, HTML report and chart generation. Linux CI repeats it offline.
The multimodal-wrapper batch-size failure was reproduced before switching to text-only loading.
Real-model CUDA execution, Docker and distributed training remain unvalidated.
