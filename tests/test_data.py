from countdown_rl.data import combination_key, split_bucket


def test_permutations_share_split():
    assert combination_key([3, 10, 7]) == combination_key([7, 3, 10])
    assert split_bucket([3, 10, 7]) == split_bucket([7, 3, 10])


def test_seed_is_deterministic():
    assert split_bucket([1, 2, 3], 42) == split_bucket([1, 2, 3], 42)

