"""A deliberately small and strict verifier for Countdown expressions."""

from __future__ import annotations

import ast
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
PLAIN_EXPR_RE = re.compile(r"^[\d\s()+\-*/]+$")


@dataclass(frozen=True)
class Verification:
    expression: str | None
    parseable: bool
    uses_numbers_exactly_once: bool
    value: Fraction | None
    correct: bool
    error: str | None = None


class InvalidExpression(ValueError):
    pass


def extract_answer(completion: str) -> str | None:
    """Return only the expression inside the final answer tag."""
    matches = ANSWER_RE.findall(completion)
    if not matches:
        return None
    expression = matches[-1].strip()
    if not expression or len(expression) > 200 or not PLAIN_EXPR_RE.fullmatch(expression):
        return None
    return expression


def _evaluate_node(node: ast.AST, leaves: list[int]) -> Fraction:
    if isinstance(node, ast.Expression):
        return _evaluate_node(node.body, leaves)
    if isinstance(node, ast.Constant) and type(node.value) is int:
        if node.value < 0:
            raise InvalidExpression("negative literals are not allowed")
        leaves.append(node.value)
        return Fraction(node.value)
    if isinstance(node, ast.UnaryOp):
        raise InvalidExpression("unary operators are not allowed")
    if not isinstance(node, ast.BinOp) or not isinstance(
        node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)
    ):
        raise InvalidExpression("only +, -, *, /, and parentheses are allowed")
    left = _evaluate_node(node.left, leaves)
    right = _evaluate_node(node.right, leaves)
    if isinstance(node.op, ast.Add):
        return left + right
    if isinstance(node.op, ast.Sub):
        return left - right
    if isinstance(node.op, ast.Mult):
        return left * right
    if right == 0:
        raise InvalidExpression("division by zero")
    return left / right


def evaluate_expression(expression: str) -> tuple[Fraction, list[int]]:
    if len(expression) > 200 or not PLAIN_EXPR_RE.fullmatch(expression):
        raise InvalidExpression("expression contains disallowed characters")
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError) as exc:
        raise InvalidExpression("expression is not valid arithmetic") from exc
    leaves: list[int] = []
    value = _evaluate_node(tree, leaves)
    return value, leaves


def verify_completion(completion: str, nums: Sequence[int], target: int) -> Verification:
    expression = extract_answer(completion)
    if expression is None:
        return Verification(None, False, False, None, False, "missing or invalid answer tag")
    try:
        value, leaves = evaluate_expression(expression)
    except InvalidExpression as exc:
        return Verification(expression, False, False, None, False, str(exc))
    exact_usage = Counter(leaves) == Counter(int(number) for number in nums)
    return Verification(
        expression=expression,
        parseable=True,
        uses_numbers_exactly_once=exact_usage,
        value=value,
        correct=exact_usage and value == target,
        error=None if exact_usage else "numbers were not used exactly once",
    )

