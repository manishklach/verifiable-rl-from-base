"""Model evaluation with pass@k, strict verification, and failure analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from datasets import Dataset

from .data import prepare_dataset
from .metrics import summarize
from .prompts import make_prompt
from .solver import difficulty_features
from .verifier import answer_block, verify_completion


def load_model(model_path: str):
    from peft import PeftConfig, PeftModel
    from transformers import AutoTokenizer

    from .modeling import load_base_model

    path = Path(model_path)
    adapter_config = path / "adapter_config.json"
    base_name = model_path
    if adapter_config.exists():
        base_name = PeftConfig.from_pretrained(model_path).base_model_name_or_path
    model = load_base_model(base_name, dtype="auto", device_map="auto")
    if adapter_config.exists():
        model = PeftModel.from_pretrained(model, model_path)
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
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    unique = []
    seen = set()
    for row in rows:
        nums, target = row["nums"], row["target"]
        if (
            not isinstance(nums, list)
            or not 1 <= len(nums) <= 5
            or any(type(n) is not int or n < 0 for n in nums)
            or type(target) is not int
        ):
            raise ValueError("JSONL puzzles require 1..5 nonnegative integers and integer target")
        identity = _problem_id(nums, target)
        if identity not in seen:
            seen.add(identity)
            unique.append(
                {
                    "nums": nums,
                    "target": target,
                    "problem_id": identity,
                    "prompt": make_prompt(nums, target),
                }
            )
    if not unique:
        raise ValueError("benchmark is empty")
    return Dataset.from_list(unique[:samples])


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
        "has_answer_tag": answer_block(completion) is not None,
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

    if min(args.samples, args.samples_per_prompt, args.max_new_tokens) < 1 or args.temperature <= 0:
        parser.error("sample counts, max-new-tokens and temperature must be positive")

    if args.jsonl:
        evaluation = _load_jsonl(args.jsonl, args.samples)
    else:
        _, evaluation = prepare_dataset(args.dataset, 1, args.samples, args.seed)
    import torch

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
            eos = tokenizer.eos_token_id
            if eos is not None and eos in completion_ids.tolist():
                completion_ids = completion_ids[: completion_ids.tolist().index(eos) + 1]
            completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
            records.append(_record(row, completion, index, len(completion_ids)))

    metrics = summarize(records)
    summary = {
        "schema_version": 2,
        "model": args.model,
        "seed": args.seed,
        "decoding": {
            "do_sample": args.samples_per_prompt > 1,
            "temperature": args.temperature if args.samples_per_prompt > 1 else None,
            "samples_per_prompt": args.samples_per_prompt,
            "max_new_tokens": args.max_new_tokens,
        },
        "dataset": {"source": args.jsonl or args.dataset, "fingerprint": evaluation._fingerprint},
        "metrics": metrics,
        "records": records,
    }
    destination = Path(args.output or f"outputs/eval-{Path(args.model).name}.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"model": args.model, **metrics}, indent=2))


if __name__ == "__main__":
    main()
