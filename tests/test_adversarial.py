from hypothesis import given
from hypothesis import strategies as st

from countdown_rl.verifier import verify_completion


@given(st.integers(min_value=1, max_value=100))
def test_target_literal_never_passes_without_matching_single_input(target):
    assert not verify_completion(f"<answer>{target}</answer>", [2, 3, 4], target).correct


def test_multiple_answer_blocks_are_rejected():
    completion = "<answer>(8 - 3) * 4</answer><answer>20</answer>"
    assert not verify_completion(completion, [8, 3, 4], 20).correct


def test_unicode_operator_is_rejected():
    assert not verify_completion("<answer>8 × 3 - 4</answer>", [8, 3, 4], 20).parseable


def test_code_injection_is_rejected():
    payload = "<answer>__import__('os').system('echo hacked')</answer>"
    assert not verify_completion(payload, [8, 3, 4], 20).parseable


def test_decimal_smuggling_is_rejected():
    assert not verify_completion("<answer>8 * 3 - 4.0</answer>", [8, 3, 4], 20).parseable
