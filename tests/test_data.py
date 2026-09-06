import pytest
from datasets import Dataset

from countdown_rl.data import assert_combination_disjoint, combination_key, split_bucket


def test_permutations_share_split():
    assert combination_key([3, 10, 7]) == combination_key([7, 3, 10])
    assert split_bucket([3, 10, 7]) == split_bucket([7, 3, 10])


def test_seed_is_deterministic():
    assert split_bucket([1, 2, 3], 42) == split_bucket([1, 2, 3], 42)


def test_disjoint_assertion_catches_permuted_combination():
    train = Dataset.from_dict({"nums": [[1, 2, 3]]})
    evaluation = Dataset.from_dict({"nums": [[3, 1, 2]]})
    with pytest.raises(AssertionError, match="leakage"):
        assert_combination_disjoint(train, evaluation)
