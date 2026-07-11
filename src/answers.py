"""Answer formatting and normalization helpers.

The dataset has two answer views: `conv_answers` are human-facing strings,
while `executed_answers` are raw program outputs. Headline evaluation uses the
raw execution result; display-normalized comparison is kept for diagnostics.
"""

from __future__ import annotations

import math
import re
from typing import Union

from pydantic import BaseModel, ConfigDict

AnswerValue = Union[float, int, str]
_NUMBER_PATTERN = re.compile(r"[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)")
_FINAL_ANSWER_PATTERN = re.compile(
    r"final\s+answer\s*:\s*(?P<answer>[^\n\r]+)",
    flags=re.IGNORECASE,
)
_PERCENT_QUESTION_PATTERN = re.compile(r"\b(?:percent|percentage|portion|ratio)\b", flags=re.IGNORECASE)


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


def parse_answer_text(answer: str, question: str | None = None) -> NormalizedAnswer:
    """Parse an answer string into a comparable value when possible."""
    answer_text = _extract_final_answer(answer)
    normalized_text = _normalize_text(answer_text)
    matches = list(_NUMBER_PATTERN.finditer(normalized_text))
    match = matches[-1] if matches else None
    if match is None:
        return NormalizedAnswer(raw=answer_text, value=normalized_text, is_numeric=False)

    value = float(match.group().replace(",", ""))
    is_percent = "%" in normalized_text or _implied_percent_answer(
        full_answer=answer,
        final_answer=answer_text,
        question=question,
    )
    if is_percent:
        # Store percentages on the same scale as ConvFinQA executed answers.
        value = value / 100

    return NormalizedAnswer(raw=answer_text, value=value, is_numeric=True, is_percent=is_percent)


def normalize_gold_answer(executed_answer: AnswerValue, conv_answer: str) -> NormalizedAnswer:
    """Normalize gold fields with the conversational answer as display aid.

    `conv_answer` preserves display intent, such as percentages. If it cannot be
    parsed numerically, fall back to `executed_answer`. Use this for diagnostic
    display-normalized scoring, not for the paper-aligned headline metric.
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


def normalize_executed_answer(executed_answer: AnswerValue) -> NormalizedAnswer:
    """Normalize the executable gold result for strict execution accuracy."""
    if isinstance(executed_answer, (float, int)):
        return NormalizedAnswer(
            raw=str(executed_answer),
            value=float(executed_answer),
            is_numeric=True,
        )

    return parse_answer_text(str(executed_answer))


def answers_match(
    prediction: str,
    executed_answer: AnswerValue,
    question: str | None = None,
    rel_tol: float = 1e-3,
    abs_tol: float = 1e-3,
) -> bool:
    """Compare a predicted answer with executable gold only."""
    parsed_prediction = parse_answer_text(prediction, question=question)
    gold_answer = normalize_executed_answer(executed_answer)

    return _normalized_answers_match(
        parsed_prediction=parsed_prediction,
        gold_answer=gold_answer,
        rel_tol=rel_tol,
        abs_tol=abs_tol,
    )


def display_answers_match(
    prediction: str,
    executed_answer: AnswerValue,
    conv_answer: str,
    question: str | None = None,
    rel_tol: float = 1e-3,
    abs_tol: float = 1e-3,
) -> bool:
    """Compare a prediction with display-normalized gold for diagnostics."""
    parsed_prediction = parse_answer_text(prediction, question=question)
    gold_answer = normalize_gold_answer(executed_answer, conv_answer)

    return _normalized_answers_match(
        parsed_prediction=parsed_prediction,
        gold_answer=gold_answer,
        rel_tol=rel_tol,
        abs_tol=abs_tol,
    )


def _normalized_answers_match(
    parsed_prediction: NormalizedAnswer,
    gold_answer: NormalizedAnswer,
    rel_tol: float,
    abs_tol: float,
) -> bool:
    """Compare two normalized answers with numeric tolerance when possible."""
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


def _extract_final_answer(text: str) -> str:
    """Prefer an explicit final-answer line when the model provides one."""
    match = _FINAL_ANSWER_PATTERN.search(text)
    if match is None:
        return text
    return match.group("answer").strip()


def _implied_percent_answer(
    full_answer: str,
    final_answer: str,
    question: str | None,
) -> bool:
    """Infer a missing percent sign from the question and calculation text."""
    # Keep this intentionally narrow: it fixes false negatives where the model
    # calculated a percentage but omitted "%" on the machine-readable final line.
    # It does not use conv_answers and it should not turn ordinary numbers into
    # percentages unless the question and calculation both support that reading.
    if "%" in final_answer:
        return False
    if question is None or _PERCENT_QUESTION_PATTERN.search(question) is None:
        return False

    normalized_answer = _normalize_text(full_answer)
    return bool(
        re.search(r"\bcalculation\s*:", normalized_answer)
        and (
            "%" in normalized_answer
            or re.search(r"(?:\*|x|×)\s*100\b", normalized_answer)
        )
    )
