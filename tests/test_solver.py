from countdown_rl.solver import difficulty_features, solve
from countdown_rl.verifier import verify_completion


def test_solver_finds_verifiable_solution():
    solution = solve([8, 3, 4], 20)
    assert solution is not None
    assert verify_completion(f"<answer>{solution.expression}</answer>", [8, 3, 4], 20).correct


def test_solver_handles_duplicates():
    solution = solve([5, 5, 2], 12)
    assert solution is not None
    assert verify_completion(f"<answer>{solution.expression}</answer>", [5, 5, 2], 12).correct


def test_solver_detects_impossible_problem():
    assert solve([1, 1, 1], 100) is None
    assert difficulty_features([1, 1, 1], 100)["difficulty_band"] == "impossible"


def test_difficulty_is_deterministic():
    assert difficulty_features([3, 8, 4], 20) == difficulty_features([4, 3, 8], 20)

