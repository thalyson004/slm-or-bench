"""Strict extraction of numeric and code outputs."""

from __future__ import annotations

import math
import re

_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_DIRECT_PATTERN = re.compile(rf"^FINAL_OBJECTIVE=\s*({_NUMBER})\s*$", re.MULTILINE)
_CODE_FENCE = re.compile(r"```(?:python)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def extract_direct_objective(text: str) -> float | None:
    matches = _DIRECT_PATTERN.findall(text)
    if len(matches) != 1:
        return None
    value = float(matches[0])
    return value if math.isfinite(value) else None


def extract_python_code(text: str) -> str:
    matches = _CODE_FENCE.findall(text)
    if matches:
        if len(matches) != 1:
            raise ValueError("The response contains multiple code blocks.")
        code = matches[0].strip()
    else:
        code = text.strip()
    if not code:
        raise ValueError("The response contains no code.")
    return code


def objectives_match(
    predicted: float | None,
    expected: float,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> bool:
    if predicted is None or not math.isfinite(predicted):
        return False
    return abs(predicted - expected) <= max(
        absolute_tolerance, relative_tolerance * abs(expected)
    )
