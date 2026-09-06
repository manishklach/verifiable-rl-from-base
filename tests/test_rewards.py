from countdown_rl.rewards import (
    correctness_reward,
    format_reward,
    number_usage_reward,
    parseable_reward,
)


def test_reward_components_are_independent():
    completions = ["Reasoning. <answer>(8 - 3) * 4</answer>"]
    kwargs = {"nums": [[8, 3, 4]], "target": [20]}
    assert format_reward(completions, **kwargs) == [0.1]
    assert parseable_reward(completions, **kwargs) == [0.25]
    assert number_usage_reward(completions, **kwargs) == [0.5]
    assert correctness_reward(completions, **kwargs) == [2.0]


def test_printing_target_cannot_earn_correctness():
    completions = ["<answer>20</answer>"]
    kwargs = {"nums": [[8, 3, 4]], "target": [20]}
    assert correctness_reward(completions, **kwargs) == [0.0]
    assert number_usage_reward(completions, **kwargs) == [0.0]
