"""Answer formatting and normalization helpers.

The dataset has two answer views: `conv_answers` are human-facing strings,
while `executed_answers` are raw program outputs. These helpers keep chat UX
natural while making later evaluation numeric and tolerant of display formats.
"""

from __future__ import annotations

import math
import re
from typing import Union

from pydantic import BaseModel, ConfigDict

AnswerValue = Union[float, int, str]
_NUMBER_PATTERN = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


class NormalizedAnswer(BaseModel):
    """Comparable answer value.

    Percentage strings are normalized to their ratio value:
    `-3.3%` becomes `-0.033`.
    """

    model_config = ConfigDict(extra="forbid")

    raw: str
    value: float | str
    is_numeric: bool
    is_percent: bool = False


def parse_answer_text(answer: str) -> NormalizedAnswer:
    """Parse an answer string into a comparable value when possible."""
    normalized_text = _normalize_text(answer)
    match = _NUMBER_PATTERN.search(normalized_text)
    if match is None:
        return NormalizedAnswer(raw=answer, value=normalized_text, is_numeric=False)

    value = float(match.group().replace(",", ""))
    is_percent = "%" in normalized_text
    if is_percent:
        # Store percentages on the same scale as ConvFinQA executed answers.
        value = value / 100

    return NormalizedAnswer(raw=answer, value=value, is_numeric=True, is_percent=is_percent)


def normalize_gold_answer(executed_answer: AnswerValue, conv_answer: str) -> NormalizedAnswer:
    """Normalize gold answer fields for later evaluation.

    `conv_answer` preserves display intent, such as percentages. If it cannot be
    parsed numerically, fall back to `executed_answer`.
    """
    parsed_conv_answer = parse_answer_text(conv_answer)
    if parsed_conv_answer.is_numeric:
        # Prefer display answer scale: "-3.3%" should compare as -0.033.
        return parsed_conv_answer

    if isinstance(executed_answer, (float, int)):
        # Some answers are not display-formatted, so the raw execution result is
        # the best comparable value.
        return NormalizedAnswer(
            raw=str(executed_answer),
            value=float(executed_answer),
            is_numeric=True,
        )

    return parse_answer_text(str(executed_answer))


def answers_match(
    prediction: str,
    executed_answer: AnswerValue,
    conv_answer: str,
    rel_tol: float = 1e-3,
    abs_tol: float = 1e-3,
) -> bool:
    """Compare a predicted answer with gold answer fields."""
    parsed_prediction = parse_answer_text(prediction)
    gold_answer = normalize_gold_answer(executed_answer, conv_answer)

    if parsed_prediction.is_numeric and gold_answer.is_numeric:
        assert isinstance(parsed_prediction.value, float)
        assert isinstance(gold_answer.value, float)
        return math.isclose(
            parsed_prediction.value,
            gold_answer.value,
            rel_tol=rel_tol,
            abs_tol=abs_tol,
        )

    return str(parsed_prediction.value).strip().lower() == str(gold_answer.value).strip().lower()


def _normalize_text(text: str) -> str:
    """Normalize common model wording before numeric parsing."""
    return (
        text.strip()
        .lower()
        .replace("negative ", "-")
        .replace("negative", "-")
        .replace("positive", "")
        .replace("−", "-")
    )
