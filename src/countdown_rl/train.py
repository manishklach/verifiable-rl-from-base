from __future__ import annotations

import json
from pathlib import Path

import torch
from peft import LoraConfig
from transformers import AutoTokenizer, set_seed
from trl import GRPOConfig, GRPOTrainer

from .config import ExperimentConfig, config_argument
from .data import prepare_dataset
from .modeling import load_base_model
from .rewards import reward_functions


def run_training(cfg: ExperimentConfig, datasets=None, reward_funcs=None) -> Path:
    set_seed(cfg.seed)
    output = Path(cfg.output_dir)
    if (
        output.exists()
        and (any(output.glob("checkpoint-*")) or (output / "final").exists())
        and not cfg.resume_from_checkpoint
    ):
        raise ValueError(
            f"Output contains training checkpoints: {output}; choose a new output_dir or resume"
        )
    output.mkdir(parents=True, exist_ok=True)
    config_suffix = cfg.difficulty_band or "main"
    (output / f"experiment_config-{config_suffix}.json").write_text(
        json.dumps(vars(cfg), indent=2) + "\n"
    )

    train_data, eval_data = (
        datasets
        if datasets is not None
        else prepare_dataset(
            cfg.dataset_name,
            cfg.train_size,
            cfg.eval_size,
            cfg.seed,
            annotate_difficulty=cfg.difficulty_band is not None,
        )
    )
    if not len(train_data) or not len(eval_data):
        raise ValueError("training and evaluation datasets must both be nonempty")
    if cfg.difficulty_band:
        train_data = train_data.filter(
            lambda row: row["difficulty_band"] == cfg.difficulty_band,
            desc=f"Selecting {cfg.difficulty_band} curriculum stage",
        )
        if not len(train_data):
            raise ValueError(f"No training examples in difficulty band {cfg.difficulty_band!r}")
    (output / f"dataset_manifest-{config_suffix}.json").write_text(
        json.dumps(
            {
                "dataset": cfg.dataset_name,
                "seed": cfg.seed,
                "train_rows": len(train_data),
                "eval_rows": len(eval_data),
                "train_fingerprint": train_data._fingerprint,
                "eval_fingerprint": eval_data._fingerprint,
                "split_rule": "sha256(seed:sorted-number-multiset) modulo 100; train < 90",
            },
            indent=2,
        )
        + "\n"
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    args = GRPOConfig(
        output_dir=cfg.output_dir,
        seed=cfg.seed,
        max_steps=cfg.max_steps,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_generations=cfg.num_generations,
        per_device_eval_batch_size=cfg.num_generations,
        ignore_data_skip=cfg.difficulty_band is not None,
        max_completion_length=cfg.max_completion_length,
        temperature=cfg.temperature,
        beta=cfg.beta,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        bf16=cfg.bf16 and torch.cuda.is_available(),
        gradient_checkpointing=cfg.gradient_checkpointing,
        report_to=[] if cfg.report_to == "none" else [cfg.report_to],
        remove_unused_columns=False,
    )
    peft_config = None
    if cfg.use_lora:
        peft_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        )

    trainer = GRPOTrainer(
        model=load_base_model(
            cfg.model_name,
            dtype=torch.bfloat16 if cfg.bf16 and torch.cuda.is_available() else torch.float32,
        ),
        reward_funcs=reward_funcs
        if reward_funcs is not None
        else reward_functions(cfg.reward_profile),
        args=args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)
    final_dir = output / "final"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    trainer.state.save_to_json(str(output / "trainer_state.json"))
    return final_dir


def main() -> None:
    run_training(ExperimentConfig.from_yaml(config_argument()))


if __name__ == "__main__":
    main()
