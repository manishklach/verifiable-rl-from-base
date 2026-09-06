"""Model evaluation with pass@k, strict verification, and failure analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import PeftConfig, PeftModel
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
)

from .data import prepare_dataset
from .metrics import summarize
from .solver import difficulty_features
from .verifier import ANSWER_RE, verify_completion


def load_model(model_path: str):
    path = Path(model_path)
    adapter_config = path / "adapter_config.json"
    base_name = model_path
    if adapter_config.exists():
        base_name = PeftConfig.from_pretrained(model_path).base_model_name_or_path
    architecture = (AutoConfig.from_pretrained(base_name, trust_remote_code=True).architectures or [""])[0]
    auto_model = (
        AutoModelForImageTextToText
        if architecture.endswith("ForConditionalGeneration")
        else AutoModelForCausalLM
    )
    if adapter_config.exists():
        peft_cfg = PeftConfig.from_pretrained(model_path)
        base = auto_model.from_pretrained(
            peft_cfg.base_model_name_or_path,
            dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, model_path)
    else:
        model = auto_model.from_pretrained(
            model_path, dtype="auto", device_map="auto", trust_remote_code=True
        )
    tokenizer_source = (
        model_path if (path / "tokenizer_config.json").exists() else model.config.name_or_path
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return model, tokenizer


def _problem_id(nums: list[int], target: int) -> str:
    value = f"{','.join(map(str, sorted(nums)))}:{target}"
    return hashlib.sha256(value.encode()).hexdigest()[:16]


def _load_jsonl(path: str, samples: int) -> Dataset:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return Dataset.from_list(rows[:samples])


def _record(row, completion: str, sample_index: int, token_count: int) -> dict[str, object]:
    result = verify_completion(completion, row["nums"], row["target"])
    features = {
        key: row[key]
        for key in ("difficulty_score", "difficulty_band", "requires_fraction")
        if key in row
    }
    if "difficulty_band" not in features:
        features.update(difficulty_features(row["nums"], row["target"]))
    return {
        "problem_id": row.get("problem_id", _problem_id(row["nums"], row["target"])),
        "sample_index": sample_index,
        "nums": row["nums"],
        "target": row["target"],
        "completion": completion,
        "completion_tokens": token_count,
        "expression": result.expression,
        "has_answer_tag": bool(ANSWER_RE.search(completion)),
        "parseable": result.parseable,
        "uses_numbers_exactly_once": result.uses_numbers_exactly_once,
        "value": str(result.value) if result.value is not None else None,
        "correct": result.correct,
        "error": result.error,
        **features,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", default="Jiayi-Pan/Countdown-Tasks-3to4")
    parser.add_argument("--jsonl", help="Evaluate a generated JSONL benchmark instead")
    parser.add_argument("--samples", type=int, default=500, help="Number of distinct problems")
    parser.add_argument("--samples-per-prompt", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.jsonl:
        evaluation = _load_jsonl(args.jsonl, args.samples)
    else:
        _, evaluation = prepare_dataset(args.dataset, 1, args.samples, args.seed)
    model, tokenizer = load_model(args.model)
    model.eval()
    records = []
    torch.manual_seed(args.seed)
    for row in evaluation:
        inputs = tokenizer(row["prompt"], return_tensors="pt").to(model.device)
        generation = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.samples_per_prompt > 1,
            "num_return_sequences": args.samples_per_prompt,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if args.samples_per_prompt > 1:
            generation["temperature"] = args.temperature
        with torch.inference_mode():
            generated = model.generate(**inputs, **generation)
        prompt_length = inputs["input_ids"].shape[1]
        for index, sequence in enumerate(generated):
            completion_ids = sequence[prompt_length:]
            completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
            records.append(_record(row, completion, index, len(completion_ids)))

    metrics = summarize(records)
    summary = {"model": args.model, "seed": args.seed, "metrics": metrics, "records": records}
    destination = Path(args.output or f"outputs/eval-{Path(args.model).name}.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"model": args.model, **metrics}, indent=2))


if __name__ == "__main__":
    main()
