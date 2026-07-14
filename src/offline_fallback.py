"""Limited offline numeric fallback for v5.

This module is intentionally small: it is not a full ConvFinQA solver. When the
LLM path is unavailable or fails to produce an executable v5 plan, it generates a
few simple candidates from selected evidence snippets, executes them with the v5
calculation-plan executor, and returns the highest-confidence answer only when a
lightweight score is decisive enough.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from src.calculation_plan import (
    CalculationPlan,
    CalculationStep,
    CalculationValue,
    execute_calculation_plan,
)
from src.evidence import EvidenceSnippet
from src.prompts import ChatTurn

_NUMBER_PATTERN = re.compile(r"[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)%?")
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_PERCENT_PATTERN = re.compile(r"\b(?:percent|percentage|portion|proportion|ratio|rate)\b", re.IGNORECASE)
_DIFFERENCE_PATTERN = re.compile(r"\b(?:difference|change|variation|increase|decrease)\b", re.IGNORECASE)
_MAXIMUM_PATTERN = re.compile(r"\b(?:maximum|max|could be issued|allowed|capacity)\b", re.IGNORECASE)
_TOTAL_PATTERN = re.compile(r"\b(?:total|facility|capacity|commitment|commitments|amount|borrowing)\b", re.IGNORECASE)
_BASE_AMOUNT_PATTERN = re.compile(r"\b(?:total|capacity|commitment|commitments|amount|borrowing)\b", re.IGNORECASE)
_ZERO_PATTERN = re.compile(r"\b(?:no|none|zero)\b", re.IGNORECASE)
_STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "then",
    "this",
    "to",
    "was",
    "were",
    "what",
    "which",
}


class OfflineFallbackResult(BaseModel):
    """Result from the limited offline fallback."""

    model_config = ConfigDict(extra="forbid")

    answered: bool
    text: str
    confidence: float = 0.0
    reason: str | None = None


class _ExtractedValue(BaseModel):
    """One numeric value and its local text context."""

    model_config = ConfigDict(extra="forbid")

    value_id: str
    label: str
    value: float
    evidence_id: str
    evidence_rank: int
    is_percent: bool = False
    has_zero_cue: bool = False
    is_temporal: bool = False


class _Candidate(BaseModel):
    """One executable offline candidate."""

    model_config = ConfigDict(extra="forbid")

    plan: CalculationPlan
    score: float
    operation: str
    values: list[_ExtractedValue]


def answer_with_offline_fallback(
    *,
    question: str,
    history: Sequence[ChatTurn],
    evidence_snippets: Sequence[EvidenceSnippet],
    confidence_threshold: float = 0.45,
) -> OfflineFallbackResult:
    """Try to answer simple numeric questions without an LLM call."""
    extracted_values = _extract_values(evidence_snippets)
    if not extracted_values:
        return OfflineFallbackResult(
            answered=False,
            text="Final answer: unable to determine with offline fallback",
            reason="No numeric values found in selected evidence.",
        )

    candidates = _generate_candidates(
        question=question,
        history=history,
        extracted_values=extracted_values,
    )
    executable_candidates = _execute_candidates(candidates)
    if not executable_candidates:
        return OfflineFallbackResult(
            answered=False,
            text="Final answer: unable to determine with offline fallback",
            reason="No executable offline candidates generated.",
        )

    scored = _with_confidences(executable_candidates)
    best_candidate, best_value, confidence = scored[0]
    if confidence < confidence_threshold:
        return OfflineFallbackResult(
            answered=False,
            text=(
                "Final answer: unable to determine with offline fallback\n"
                f"Offline fallback confidence: {confidence:.2f}"
            ),
            confidence=confidence,
            reason="Top offline candidate confidence below threshold.",
        )

    return OfflineFallbackResult(
        answered=True,
        text=_format_candidate_answer(best_candidate, best_value, confidence),
        confidence=confidence,
    )


def _extract_values(evidence_snippets: Sequence[EvidenceSnippet]) -> list[_ExtractedValue]:
    values: list[_ExtractedValue] = []
    for evidence_rank, snippet in enumerate(evidence_snippets):
        for number_index, match in enumerate(_NUMBER_PATTERN.finditer(snippet.text), start=1):
            raw_number = match.group()
            parsed_value = _parse_number(raw_number)
            if parsed_value is None:
                continue
            label = _label_for_number(snippet.text, match.start(), match.end())
            values.append(
                _ExtractedValue(
                    value_id=_value_id(label, snippet.snippet_id, number_index),
                    label=label,
                    value=parsed_value,
                    evidence_id=snippet.snippet_id,
                    evidence_rank=evidence_rank,
                    is_percent="%" in raw_number,
                    has_zero_cue=_has_zero_cue(snippet.text, match.start()),
                    is_temporal=_is_temporal_value(parsed_value, snippet.text, match.start(), match.end()),
                ),
            )
    return values


def _generate_candidates(
    *,
    question: str,
    history: Sequence[ChatTurn],
    extracted_values: Sequence[_ExtractedValue],
) -> list[_Candidate]:
    candidates = [_select_candidate(question, value) for value in extracted_values]
    if _PERCENT_PATTERN.search(question):
        candidates.extend(_ratio_candidates(question, extracted_values))
    if _DIFFERENCE_PATTERN.search(question):
        candidates.extend(_difference_candidates(question, extracted_values))
    if _MAXIMUM_PATTERN.search(question):
        candidates.extend(_multiply_candidates(question, extracted_values))
    if history and _PERCENT_PATTERN.search(question):
        candidates.extend(_history_ratio_candidates(question, history, extracted_values))
    return candidates


def _select_candidate(question: str, value: _ExtractedValue) -> _Candidate:
    score = 1.0 + _label_overlap_score(question, value.label) + _evidence_rank_score(value)
    score += _domain_match_score(question, value)
    if _is_suspicious_zero_value(value) and not _question_asks_for_zero(question):
        score -= 1.5
    if value.is_temporal and not _question_asks_for_date(question):
        score -= 1.5
    return _Candidate(
        plan=CalculationPlan(
            values=[_plan_value(value)],
            steps=[CalculationStep(id="answer", op="select", args=[value.value_id])],
            answer="answer",
        ),
        score=score,
        operation="select",
        values=[value],
    )


def _ratio_candidates(question: str, values: Sequence[_ExtractedValue]) -> list[_Candidate]:
    candidates: list[_Candidate] = []
    for numerator in values:
        for denominator in values:
            if numerator == denominator or math.isclose(denominator.value, 0.0, abs_tol=1e-12):
                continue
            if math.isclose(numerator.value, denominator.value, rel_tol=0.0, abs_tol=1e-12):
                continue
            score = 1.0 + _label_overlap_score(question, numerator.label) + _evidence_rank_score(numerator)
            score += _domain_match_score(question, numerator)
            score += 1.0 if _TOTAL_PATTERN.search(denominator.label) else -0.3
            score += 0.5 if not _TOTAL_PATTERN.search(numerator.label) else -1.0
            if _is_suspicious_zero_value(numerator) and not _question_asks_for_zero(question):
                score -= 1.5
            if numerator.is_temporal or denominator.is_temporal:
                score -= 1.0
            candidates.append(
                _Candidate(
                    plan=CalculationPlan(
                        values=[_plan_value(numerator), _plan_value(denominator)],
                        steps=[CalculationStep(id="answer", op="divide", args=[numerator.value_id, denominator.value_id])],
                        answer="answer",
                    ),
                    score=score,
                    operation="divide",
                    values=[numerator, denominator],
                ),
            )
    return candidates


def _difference_candidates(question: str, values: Sequence[_ExtractedValue]) -> list[_Candidate]:
    candidates: list[_Candidate] = []
    for left in values[:8]:
        for right in values[:8]:
            if left == right:
                continue
            score = 0.7 + _label_overlap_score(question, left.label) + _label_overlap_score(question, right.label)
            score += _evidence_rank_score(left) + _evidence_rank_score(right)
            if left.is_temporal or right.is_temporal:
                score -= 1.0
            candidates.append(
                _Candidate(
                    plan=CalculationPlan(
                        values=[_plan_value(left), _plan_value(right)],
                        steps=[CalculationStep(id="answer", op="subtract", args=[left.value_id, right.value_id])],
                        answer="answer",
                    ),
                    score=score,
                    operation="subtract",
                    values=[left, right],
                ),
            )
    return candidates


def _multiply_candidates(question: str, values: Sequence[_ExtractedValue]) -> list[_Candidate]:
    candidates: list[_Candidate] = []
    for amount in values:
        if _MAXIMUM_PATTERN.search(question) and not _BASE_AMOUNT_PATTERN.search(amount.label):
            continue
        for rate in values:
            if amount == rate:
                continue
            rate_value = rate.value / 100 if rate.is_percent else rate.value
            if not 0 < abs(rate_value) <= 1:
                continue
            score = 1.0 + _label_overlap_score(question, amount.label)
            score += _domain_match_score(question, amount)
            score += 1.0 if _TOTAL_PATTERN.search(amount.label) else 0.0
            score += 0.7 if rate.is_percent else 0.0
            if amount.is_temporal:
                score -= 1.0
            candidates.append(
                _Candidate(
                    plan=CalculationPlan(
                        values=[
                            _plan_value(amount),
                            CalculationValue(id=rate.value_id, value=rate_value, evidence=rate.evidence_id),
                        ],
                        steps=[CalculationStep(id="answer", op="multiply", args=[amount.value_id, rate.value_id])],
                        answer="answer",
                    ),
                    score=score,
                    operation="multiply",
                    values=[amount, rate],
                ),
            )
    return candidates


def _history_ratio_candidates(
    question: str,
    history: Sequence[ChatTurn],
    values: Sequence[_ExtractedValue],
) -> list[_Candidate]:
    history_values = _history_values(history)
    if not history_values:
        return []
    return _ratio_candidates(question, [*history_values, *values])


def _execute_candidates(candidates: Sequence[_Candidate]) -> list[tuple[_Candidate, float]]:
    executed: list[tuple[_Candidate, float]] = []
    seen: set[tuple[str, float]] = set()
    for candidate in candidates:
        try:
            value = execute_calculation_plan(candidate.plan)
        except (ValueError, ZeroDivisionError):
            continue
        key = (candidate.operation, round(value, 10))
        if key in seen:
            continue
        seen.add(key)
        executed.append((candidate, value))
    return executed


def _with_confidences(executed_candidates: Sequence[tuple[_Candidate, float]]) -> list[tuple[_Candidate, float, float]]:
    ranked_candidates = sorted(executed_candidates, key=lambda item: -item[0].score)[:5]
    max_score = max(candidate.score for candidate, _value in ranked_candidates)
    weights = [math.exp(candidate.score - max_score) for candidate, _value in ranked_candidates]
    weight_sum = sum(weights)
    scored = [
        (candidate, value, weight / weight_sum)
        for (candidate, value), weight in zip(ranked_candidates, weights)
    ]
    scored.sort(key=lambda item: (-item[2], -item[0].score))
    return scored


def _format_candidate_answer(candidate: _Candidate, value: float, confidence: float) -> str:
    plan_json = candidate.plan.model_dump(exclude_none=True)
    return "\n".join(
        [
            "Target: offline numeric fallback",
            "Values:",
            *[
                f"- {value_item.value_id} = {value_item.value} from evidence {value_item.evidence_id}"
                for value_item in candidate.values
            ],
            f"Operation: {candidate.operation}",
            "",
            "Calculation plan JSON:",
            json.dumps(plan_json, separators=(",", ":")),
            "",
            f"Final answer: {_format_number(value)}",
            f"Offline fallback confidence: {confidence:.2f}",
        ],
    )


def _plan_value(value: _ExtractedValue) -> CalculationValue:
    return CalculationValue(id=value.value_id, value=value.value, evidence=value.evidence_id)


def _parse_number(raw_number: str) -> float | None:
    try:
        return float(raw_number.replace(",", "").replace("%", ""))
    except ValueError:
        return None


def _label_for_number(text: str, start: int, end: int) -> str:
    before_tokens = _TOKEN_PATTERN.findall(text[max(0, start - 80) : start].lower())
    after_tokens = _TOKEN_PATTERN.findall(text[end : min(len(text), end + 100)].lower())
    # Financial prose often writes "$4.7 million of letters of credit", where
    # the meaningful label follows the number.
    if after_tokens[:2] in (["million", "of"], ["millions", "of"]) or "of" in after_tokens[:3]:
        label_tokens = after_tokens[:12] + before_tokens[-4:]
    else:
        label_tokens = before_tokens[-8:] + after_tokens[:8]
    label_tokens = [token for token in label_tokens if token not in {"million", "millions", "percent"}]
    return " ".join(label_tokens)[:80] or "value"


def _value_id(label: str, evidence_id: str, number_index: int) -> str:
    tokens = [token for token in _TOKEN_PATTERN.findall(label.lower()) if token not in _STOPWORDS]
    stem = "_".join(tokens[:5]) or "value"
    return f"{stem}_{evidence_id.lower().replace('-', '_')}_{number_index}"


def _label_overlap_score(question: str, label: str) -> float:
    question_tokens = _tokens(question)
    label_tokens = _tokens(label)
    if not question_tokens or not label_tokens:
        return 0.0
    return min(len(question_tokens & label_tokens) * 0.5, 2.0)


def _evidence_rank_score(value: _ExtractedValue) -> float:
    return max(0.0, 1.0 - 0.15 * value.evidence_rank)


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in _TOKEN_PATTERN.findall(text.lower())
        if len(token) > 1 and token not in _STOPWORDS
    }


def _has_zero_cue(text: str, number_start: int) -> bool:
    return bool(_ZERO_PATTERN.search(text[max(0, number_start - 24) : number_start + 24]))


def _question_asks_for_zero(question: str) -> bool:
    return bool(re.search(r"\b(?:zero|no|cash borrowings|none)\b", question, flags=re.IGNORECASE))


def _is_suspicious_zero_value(value: _ExtractedValue) -> bool:
    label_tokens = _tokens(value.label)
    return math.isclose(value.value, 0.0, abs_tol=1e-12) or {"cash", "borrowings"} <= label_tokens


def _question_asks_for_date(question: str) -> bool:
    return bool(re.search(r"\b(?:what year|which year|what date|which date|when)\b", question, flags=re.IGNORECASE))


def _is_temporal_value(value: float, text: str, start: int, end: int) -> bool:
    if value.is_integer() and 1900 <= value <= 2100:
        return True
    nearby = text[max(0, start - 24) : min(len(text), end + 24)]
    return bool(re.search(r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b", nearby, flags=re.IGNORECASE))


def _domain_match_score(question: str, value: _ExtractedValue) -> float:
    question_tokens = _tokens(question)
    label_tokens = _tokens(value.label)
    score = 0.0
    drawn_question = bool({"drawn", "outstanding"} & question_tokens)
    if drawn_question and {"letters", "outstanding"} <= label_tokens:
        score += 2.0
    if drawn_question and {"facility", "amount"} <= label_tokens and "letters" not in label_tokens:
        score -= 3.0
    if not drawn_question and {"facility", "amount", "credit"} <= question_tokens and {"facility", "amount"} <= label_tokens:
        score += 2.0
    elif not drawn_question and {"facility", "credit"} <= question_tokens and "amount" in label_tokens:
        score += 1.0
    if {"maximum", "max", "issued", "capacity"} & question_tokens and {"facility", "capacity"} & label_tokens:
        score += 1.0
    return score


def _history_values(history: Sequence[ChatTurn]) -> list[_ExtractedValue]:
    values: list[_ExtractedValue] = []
    for history_index, turn in enumerate(reversed(history[-3:]), start=1):
        matches = list(_NUMBER_PATTERN.finditer(turn.assistant))
        if not matches:
            continue
        value = _parse_number(matches[-1].group())
        if value is None:
            continue
        values.append(
            _ExtractedValue(
                value_id=f"history_answer_{history_index}",
                label=f"previous answer {history_index}",
                value=value,
                evidence_id="history",
                evidence_rank=0,
            ),
        )
    return values


def _format_number(value: float) -> str:
    if math.isclose(value, round(value), rel_tol=0.0, abs_tol=1e-10):
        return str(int(round(value)))
    return f"{value:.10g}"
