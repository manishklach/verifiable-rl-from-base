.PHONY: install test lint train evaluate plot report demo ood

install:
	python -m pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check .

train:
	accelerate launch -m countdown_rl.train --config configs/qwen35-0.8b-grpo.yaml

evaluate:
	python -m countdown_rl.evaluate --model outputs/qwen35-0.8b-countdown/final

plot:
	python -m countdown_rl.plot --state outputs/qwen35-0.8b-countdown/trainer_state.json

ood:
	python -m countdown_rl.generate --arity 5 --samples 500

report:
	bash scripts/evaluate_checkpoints.sh

demo:
	python -m countdown_rl.demo --trained outputs/qwen35-0.8b-countdown/final
