from __future__ import annotations

from collections.abc import Sequence


def make_prompt(nums: Sequence[int], target: int) -> str:
    rendered = ", ".join(str(int(number)) for number in nums)
    return f"""Solve this Countdown arithmetic puzzle.

Numbers: [{rendered}]
Target: {int(target)}

Use every supplied number exactly once. You may use +, -, *, /, and parentheses.
Do not introduce any other numbers. Explain your search briefly, then put only the
final arithmetic expression inside <answer>...</answer>.

Solution:
"""

