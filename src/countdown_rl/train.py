from __future__ import annotations

import json
from pathlib import Path

import torch
from peft import LoraConfig
from transformers import AutoTokenizer, set_seed
from trl import GRPOConfig, GRPOTrainer

from .config import ExperimentConfig, config_argument
from .data import prepare_dataset
from .rewards import REWARD_FUNCTIONS


def main() -> None:
    cfg = ExperimentConfig.from_yaml(config_argument())
    set_seed(cfg.seed)
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "experiment_config.json").write_text(json.dumps(vars(cfg), indent=2) + "\n")

    train_data, eval_data = prepare_dataset(
        cfg.dataset_name, cfg.train_size, cfg.eval_size, cfg.seed
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
        max_prompt_length=cfg.max_prompt_length,
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
        model=cfg.model_name,
        reward_funcs=REWARD_FUNCTIONS,
        args=args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    final_dir = output / "final"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    trainer.state.save_to_json(str(output / "trainer_state.json"))


if __name__ == "__main__":
    main()
