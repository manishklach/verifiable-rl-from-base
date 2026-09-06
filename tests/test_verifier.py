import pytest

from countdown_rl.verifier import InvalidExpression, evaluate_expression, verify_completion


def test_accepts_correct_expression():
    result = verify_completion("work\n<answer>(8 - 3) * 4</answer>", [8, 3, 4], 20)
    assert result.correct


def test_uses_repeated_numbers_exactly():
    assert verify_completion("<answer>5 + 5 + 2</answer>", [5, 5, 2], 12).correct
    assert not verify_completion("<answer>5 + 2 + 2</answer>", [5, 5, 2], 9).correct


def test_rejects_target_literal_hack():
    result = verify_completion("The answer is 42. <answer>42</answer>", [6, 7, 3], 42)
    assert result.parseable
    assert not result.uses_numbers_exactly_once
    assert not result.correct


@pytest.mark.parametrize(
    "expression",
    ["2 ** 3", "abs(2)", "__import__('os')", "2 // 1", "-2 + 4"],
)
def test_rejects_disallowed_syntax(expression):
    with pytest.raises(InvalidExpression):
        evaluate_expression(expression)


def test_exact_fraction_arithmetic():
    result = verify_completion("<answer>8 / (3 - 1)</answer>", [8, 3, 1], 4)
    assert result.correct


def test_requires_answer_tag():
    assert not verify_completion("(8 - 3) * 4", [8, 3, 4], 20).correct
