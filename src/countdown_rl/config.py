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
    reward_profile: str = "shaped"
    difficulty_band: str | None = None
    resume_from_checkpoint: bool | str = False

    def __post_init__(self):
        for name in (
            "train_size",
            "eval_size",
            "max_steps",
            "per_device_train_batch_size",
            "gradient_accumulation_steps",
            "num_generations",
            "max_completion_length",
            "save_steps",
            "eval_steps",
            "logging_steps",
            "lora_r",
            "lora_alpha",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.num_generations < 2:
            raise ValueError("num_generations must be at least two")
        if self.learning_rate <= 0 or self.temperature <= 0 or self.beta < 0:
            raise ValueError("learning_rate and temperature must be positive; beta nonnegative")
        if not 0 <= self.lora_dropout < 1:
            raise ValueError("lora_dropout must be in [0, 1)")
        if self.reward_profile not in {"shaped", "binary", "no-format"}:
            raise ValueError("unknown reward_profile")
        if self.difficulty_band not in {None, "easy", "medium", "hard"}:
            raise ValueError("unknown difficulty_band")

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        values = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError("configuration must be a YAML mapping")  # noqa: TRY004
        allowed = {field.name for field in fields(cls)}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        return cls(**values)


def config_argument() -> str:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qwen35-0.8b-grpo.yaml")
    return parser.parse_args().config
