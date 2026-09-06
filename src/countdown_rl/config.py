from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
from pathlib import Path

import yaml


@dataclass
class ExperimentConfig:
    model_name: str = "Qwen/Qwen3.5-0.8B-Base"
    dataset_name: str = "Jiayi-Pan/Countdown-Tasks-3to4"
    output_dir: str = "outputs/qwen35-0.8b-countdown"
    seed: int = 42
    train_size: int = 50000
    eval_size: int = 1000
    max_steps: int = 500
    learning_rate: float = 1e-5
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    num_generations: int = 8
    max_prompt_length: int = 256
    max_completion_length: int = 256
    temperature: float = 1.0
    beta: float = 0.04
    save_steps: int = 50
    eval_steps: int = 50
    logging_steps: int = 1
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bf16: bool = True
    gradient_checkpointing: bool = True
    report_to: str = "none"

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        values = yaml.safe_load(Path(path).read_text())
        allowed = {field.name for field in fields(cls)}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        return cls(**values)


def config_argument() -> str:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qwen35-0.8b-grpo.yaml")
    return parser.parse_args().config

