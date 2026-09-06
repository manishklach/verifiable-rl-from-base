"""Execute real training, resume, adapter reload and evaluation before a long run."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path


def tiny_model(destination: Path) -> str:
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen3_5Config, Qwen3_5ForConditionalGeneration

    vocab = {"[PAD]": 0, "[EOS]": 1, "[UNK]": 2}
    vocab.update({str(i): i + 3 for i in range(61)})
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", eos_token="[EOS]", unk_token="[UNK]"
    )
    tokenizer.model_input_names = ["input_ids", "attention_mask"]
    config = Qwen3_5Config(
        text_config={
            "vocab_size": 64,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 32,
            "linear_key_head_dim": 16,
            "linear_value_head_dim": 16,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 2,
            "layer_types": ["linear_attention", "full_attention"],
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000,
                "partial_rotary_factor": 1.0,
                "mrope_section": [4, 6, 6],
            },
            "pad_token_id": 0,
            "eos_token_id": 1,
        },
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 2,
            "out_hidden_size": 64,
            "num_position_embeddings": 16,
        },
        image_token_id=60,
        video_token_id=61,
        vision_start_token_id=62,
        vision_end_token_id=63,
        pad_token_id=0,
        eos_token_id=1,
    )
    model = Qwen3_5ForConditionalGeneration(config)
    model.save_pretrained(destination)
    tokenizer.save_pretrained(destination)
    import torch

    from .modeling import load_base_model

    loaded = load_base_model(str(destination))
    for name, tensor in model.model.language_model.state_dict().items():
        assert torch.equal(tensor, loaded.model.state_dict()[name]), f"Text weight mismatch: {name}"
    assert torch.equal(model.lm_head.weight, loaded.lm_head.weight)
    return str(destination.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--tiny", action="store_true", help="Offline random tiny Qwen hybrid model")
    parser.add_argument("--output", default="outputs/hardware-smoke")
    args = parser.parse_args()
    import torch
    from datasets import Dataset

    from .config import ExperimentConfig
    from .prompts import make_prompt
    from .train import run_training

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable. Use --device cpu --tiny for an offline integration test")
    if args.device == "cpu" and not args.tiny:
        parser.error("CPU smoke requires --tiny; full-model acceptance requires a CUDA GPU")
    if args.device == "cpu" and torch.cuda.is_available():
        parser.error("Set CUDA_VISIBLE_DEVICES='' to force the CPU integration test")
    torch.set_num_threads(2)
    torch.manual_seed(42)
    root = Path(args.output).resolve()
    if root.exists():
        parser.error(f"Output already exists: {root}. Choose a new --output to preserve evidence")
    root.mkdir(parents=True)
    model_name = tiny_model(root / "tiny-base") if args.tiny else ExperimentConfig().model_name
    rows = [
        {"nums": nums, "target": target, "prompt": make_prompt(nums, target)}
        for nums, target in [([8, 3, 4], 20), ([5, 5, 2], 12), ([2, 3, 4], 9), ([6, 2, 3], 12)]
    ]
    if args.tiny:
        datasets = (Dataset.from_list(rows[:2]), Dataset.from_list(rows[2:]))
    else:
        from .data import prepare_dataset

        datasets = prepare_dataset(ExperimentConfig().dataset_name, 2, 2)
        rows = list(datasets[0]) + list(datasets[1])
    cfg = ExperimentConfig(
        model_name=model_name,
        output_dir=str(root / "training"),
        train_size=2,
        eval_size=2,
        max_steps=1,
        save_steps=1,
        eval_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        num_generations=2,
        max_completion_length=8 if args.tiny else 32,
        lora_r=2 if args.tiny else 8,
        lora_alpha=4 if args.tiny else 16,
        bf16=args.device == "cuda" and torch.cuda.is_bf16_supported(),
    )
    from .rewards import reward_functions

    rewards = reward_functions(cfg.reward_profile)
    if args.tiny:
        # Random text will not earn arithmetic rewards. Exercise nonzero policy gradients
        # with an explicit integration-only reward; never used in real experiments.
        def diagnostic_reward(completions, **kwargs):
            return [float(index % 2) for index in range(len(completions))]

        rewards = rewards + [diagnostic_reward]
    run_training(cfg, datasets=datasets, reward_funcs=rewards)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    checkpoint = root / "training" / "checkpoint-1"
    if not (checkpoint / "trainer_state.json").exists():
        raise RuntimeError("Training did not save checkpoint-1")
    final = run_training(
        replace(cfg, max_steps=2, resume_from_checkpoint=str(checkpoint)),
        datasets=datasets,
        reward_funcs=rewards,
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    puzzles = root / "puzzles.jsonl"
    puzzles.write_text("".join(json.dumps(row) + "\n" for row in rows[2:]), encoding="utf-8")
    result = root / "evaluation.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "countdown_rl.evaluate",
            "--model",
            str(final),
            "--jsonl",
            str(puzzles),
            "--samples",
            "2",
            "--max-new-tokens",
            "8",
            "--output",
            str(result),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "countdown_rl.report",
            str(result),
            "--output",
            str(root / "report.html"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "countdown_rl.plot",
            "--state",
            str(root / "training" / "trainer_state.json"),
            "--output",
            str(root / "curves.png"),
        ],
        check=True,
    )
    state = json.loads((root / "training" / "trainer_state.json").read_text())
    evaluation = json.loads(result.read_text())
    assert state["global_step"] == 2, "Resume did not reach step 2"
    assert len(evaluation["records"]) == 2, "Evaluation did not produce two records"
    if args.tiny:
        from safetensors.torch import load_file

        weights = load_file(str(final / "adapter_model.safetensors"))
        assert any(
            tensor.abs().sum() > 0 for name, tensor in weights.items() if "lora_B" in name
        ), "No adapter update"
    evidence = {
        "status": "passed",
        "tiny": args.tiny,
        "diagnostic_reward": args.tiny,
        "device": args.device,
        "model": model_name,
        "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "trl", "peft", "accelerate", "datasets")
        },
        "steps": state["global_step"],
        "evaluated_problems": 2,
    }
    (root / "acceptance.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
