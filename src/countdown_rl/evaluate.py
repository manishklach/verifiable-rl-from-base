from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import prepare_dataset
from .verifier import verify_completion


def load_model(model_path: str):
    path = Path(model_path)
    adapter_config = path / "adapter_config.json"
    if adapter_config.exists():
        peft_cfg = PeftConfig.from_pretrained(model_path)
        base = AutoModelForCausalLM.from_pretrained(
            peft_cfg.base_model_name_or_path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype="auto", device_map="auto", trust_remote_code=True
        )
    tokenizer_source = model_path if (path / "tokenizer_config.json").exists() else model.config.name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", default="Jiayi-Pan/Countdown-Tasks-3to4")
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    _, evaluation = prepare_dataset(args.dataset, 1, args.samples, args.seed)
    model, tokenizer = load_model(args.model)
    model.eval()
    records = []
    for row in evaluation:
        inputs = tokenizer(row["prompt"], return_tensors="pt").to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        completion = tokenizer.decode(
            generated[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        result = verify_completion(completion, row["nums"], row["target"])
        records.append(
            {
                "nums": row["nums"],
                "target": row["target"],
                "completion": completion,
                "expression": result.expression,
                "valid": result.parseable and result.uses_numbers_exactly_once,
                "correct": result.correct,
            }
        )
    summary = {
        "model": args.model,
        "samples": len(records),
        "valid_rate": sum(row["valid"] for row in records) / len(records),
        "accuracy": sum(row["correct"] for row in records) / len(records),
        "records": records,
    }
    destination = Path(args.output or f"outputs/eval-{Path(args.model).name}.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
