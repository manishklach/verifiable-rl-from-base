"""Fail-fast environment and configuration checks before an expensive run."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os

import torch
from transformers import AutoConfig, AutoTokenizer

from .config import ExperimentConfig
from .rewards import reward_functions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qwen35-0.8b-grpo.yaml")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    cfg = ExperimentConfig.from_yaml(args.config)
    problems = []
    if not torch.cuda.is_available() and not args.allow_cpu:
        problems.append("CUDA is unavailable; GRPO is not practical on this machine")
    if cfg.num_generations < 2:
        problems.append("GRPO requires at least two generations for a relative advantage")
    effective = (
        int(os.environ.get("WORLD_SIZE", "1"))
        * cfg.per_device_train_batch_size
        * cfg.gradient_accumulation_steps
    )
    if effective % cfg.num_generations:
        problems.append(
            "effective batch size must be divisible by num_generations "
            f"({effective} is not divisible by {cfg.num_generations})"
        )
    if cfg.bf16 and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        problems.append("BF16 is configured but unsupported; set bf16: false")
    reward_functions(cfg.reward_profile)
    model_config = AutoConfig.from_pretrained(cfg.model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    summary = {
        "model": cfg.model_name,
        "architecture": model_config.architectures,
        "vocabulary_size": len(tokenizer),
        "cuda": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "bf16_supported": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        "effective_prompt_batch": effective,
        "generations_per_prompt": cfg.num_generations,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "trl", "peft", "datasets")
        },
        "problems": problems,
    }
    print(json.dumps(summary, indent=2))
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
