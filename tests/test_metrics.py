import pytest

from countdown_rl.metrics import pass_at_k, summarize


def _record(problem, sample, correct, parseable=True, usage=True):
    return {
        "problem_id": problem,
        "sample_index": sample,
        "nums": [1, 2, 3],
        "completion": "x",
        "has_answer_tag": True,
        "parseable": parseable,
        "uses_numbers_exactly_once": usage,
        "correct": correct,
        "difficulty_band": "easy",
    }


def test_pass_at_k_groups_samples_by_problem():
    rows = [_record("a", 0, False), _record("a", 1, True), _record("b", 0, False)]
    assert pass_at_k(rows, 1) == 0.25
    with pytest.raises(ValueError, match="at least k"):
        pass_at_k(rows, 2)
    rows.append(_record("b", 1, False))
    assert pass_at_k(rows, 2) == 0.5
    assert pass_at_k(list(reversed(rows)), 1) == pass_at_k(rows, 1)


def test_summary_reports_slices():
    metrics = summarize([_record("a", 0, True), _record("b", 0, False)])
    assert metrics["strict_accuracy"] == 0.5
    assert metrics["accuracy_by_arity"] == {"3": 0.5}
