# Changelog

## 0.3.0 - 2026-09-06

- Fix Qwen GRPO scoring failure with a shared text-only checkpoint loader.
- Add constrained Linux CPU/CUDA setup and one-command experiment orchestration.
- Add tiny hybrid integration: exact text weights, nonzero adapter updates, resume,
  adapter reload, evaluation, report and plot. Run it in Linux CI.
- Add real-model CUDA acceptance with source data and normal rewards.
- Prevent accidental checkpoint overwrite; fail CPU preflight before network access.
- Document the setup path and outstanding GPU validation.


## 0.2.0 — 2026-09-06

### Correctness and experimental integrity

- Reject stray/nested answer tags consistently in verification and format rewards.
- Fail on mismatched reward metadata; report wrong-value errors explicitly.
- Deduplicate held-out puzzles by input multiset and target.
- Replace order-dependent prefix hits with the combinatorial pass@k estimator.
- Rebuild JSONL prompts, identities and difficulty rather than trusting annotations.
- Record decoding metadata, dataset fingerprint and EOS-trimmed token counts.
- Validate configuration, generation and solver inputs; bound solver cache to 16 entries.

### Training and packaging

- Remove obsolete max_prompt_length from the TRL configuration path.
- Respect bf16=false when selecting model dtype, set evaluation generation batch explicitly,
  and disable old-data skipping when resuming curriculum stages.
- Use WORLD_SIZE in doctor and flag unsupported BF16 hardware.
- Separate CPU dependencies from train, plot and demo extras; target TRL 0.29.x.
- Add Windows/Linux CI matrix, package builds and generation smoke tests.
- Add architecture, evaluation, troubleshooting and release documentation.

### Compatibility and validation boundary

Install `.[train,plot]` for the prior full experiment workflow. Remove max_prompt_length from
custom YAML. pass@k and duplicate handling intentionally differ from 0.1. Solver arity is
bounded at five. No trained weights, GPU benchmark, or improvement in model accuracy is
claimed in this release.

## 0.1.0

Initial imported framework: exact verification and solving, GRPO/LoRA training, grouped
splits, curricula, reward ablations, evaluation, reports and demo.
