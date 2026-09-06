# Troubleshooting

| Symptom | Action |
|---|---|
| Missing torch, TRL or PEFT | Install `pip install -e ".[train]"` in the active environment. |
| Missing matplotlib / gradio | Install `.[plot]` / `.[train,demo]`. |
| CUDA unavailable | Use CPU for verification/tests; provision a CUDA GPU for training. |
| BF16 unsupported | Set `bf16: false`; model loading then uses float32, which uses more memory. |
| Batch divisibility error | Make processes × batch × accumulation divisible by generations. |
| Unknown YAML key | Check spelling; removed `max_prompt_length` is no longer accepted. |
| No examples in curriculum band | Increase train_size or inspect difficulty distribution. |
| OOD generation slow | Use three or four numbers, smaller ranges, and fewer puzzles. |
| No checkpoints found | Train first or pass the correct run directory to the checkpoint script. |
| Metrics differ from v0.1 | Evaluation now deduplicates puzzles and uses estimated pass@k. |

## CPU development on Windows

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"
.venv/Scripts/python -m pytest -q
.venv/Scripts/countdown-generate --samples 3 --arity 3 --maximum 10 --output outputs/smoke.jsonl
```

Bash orchestration scripts require Linux, WSL or another Bash environment. Windows CPU tests
are supported; Windows GPU training is not validated by this release. No model download is
needed for the unit suite. `countdown-doctor` does fetch model/tokenizer metadata.

## GPU acceptance checklist

Run doctor, then two training steps with short completions and a tiny split. Confirm a saved
adapter reloads, baseline and adapter evaluation complete, trainer state produces a plot, and
a resumed curriculum stage consumes new data. Keep logs and package versions. This release's
CPU validation does not replace that GPU acceptance run.
