import pytest
from datasets import Dataset

from countdown_rl.config import ExperimentConfig
from countdown_rl.data import prepare_dataset
from countdown_rl.generate import generate_benchmark
from countdown_rl.rewards import correctness_reward, format_reward
from countdown_rl.solver import solve
from countdown_rl.verifier import verify_completion


@pytest.mark.parametrize(
    "text",
    [
        "<answer>8+3+4</answer></answer>",
        "<answer><answer>8+3+4</answer>",
        "<answer>8+3+4</answer><answer>",
    ],
)
def test_unbalanced_tags_cannot_earn_reward(text):
    assert format_reward([text]) == [0.0]
    assert not verify_completion(text, [8, 3, 4], 15).correct


def test_wrong_value_has_diagnostic():
    assert verify_completion("<answer>8+3+4</answer>", [8, 3, 4], 20).error


def test_reward_rejects_misaligned_metadata():
    with pytest.raises(ValueError):
        correctness_reward(["<answer>1</answer>"], [], [])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_generations": 1},
        {"max_steps": 0},
        {"train_size": -1},
        {"lora_dropout": 1},
        {"reward_profile": "typo"},
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        ExperimentConfig(**kwargs)


def test_empty_yaml(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("")
    with pytest.raises(ValueError, match="mapping"):
        ExperimentConfig.from_yaml(path)


@pytest.mark.parametrize("nums", [[], [-1, 2, 3], [1.5, 2, 3], [True, 2, 3], [1] * 6])
def test_solver_rejects_invalid_inputs(nums):
    with pytest.raises(ValueError):
        solve(nums, 6)


def test_fraction_required_solution():
    result = solve([3, 3, 8, 8], 24)
    assert result.requires_fraction
    assert verify_completion(f"<answer>{result.expression}</answer>", [3, 3, 8, 8], 24).correct


def test_generation_is_deterministic_and_verifiable():
    first = generate_benchmark(4, 3, 1, 10, 7)
    assert first == generate_benchmark(4, 3, 1, 10, 7)
    for row in first:
        assert verify_completion(
            f"<answer>{row['reference_expression']}</answer>", row["nums"], row["target"]
        ).correct


def test_dataset_deduplicates_evaluation(monkeypatch):
    from countdown_rl.data import split_bucket

    nums = next([1, 2, n] for n in range(3, 100) if split_bucket([1, 2, n]) >= 90)
    raw = Dataset.from_list(
        [
            {"nums": nums, "target": 10},
            {"nums": list(reversed(nums)), "target": 10},
            {"nums": nums, "target": 11},
        ]
    )
    monkeypatch.setattr("countdown_rl.data.load_dataset", lambda *a, **kw: raw)
    _, evaluation = prepare_dataset("offline")
    assert len(evaluation) == 2


def test_all_shipped_configs_load():
    from pathlib import Path

    for path in Path("configs").rglob("*.yaml"):
        ExperimentConfig.from_yaml(path)


def test_jsonl_rebuilds_prompt_and_identity(tmp_path):
    import json

    from countdown_rl.evaluate import _load_jsonl

    path = tmp_path / "puzzles.jsonl"
    rows = [
        {
            "nums": [8, 3, 4],
            "target": 20,
            "prompt": "untrusted solution",
            "problem_id": "a",
            "difficulty_band": "fake",
        },
        {"nums": [4, 3, 8], "target": 20, "problem_id": "b"},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows))
    data = _load_jsonl(str(path), 10)
    assert len(data) == 1
    assert data[0]["prompt"] != "untrusted solution"
    assert "difficulty_band" not in data.column_names
