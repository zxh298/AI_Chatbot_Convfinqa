"""Lightweight no-gold verification for model answers.

The verifier is used by the v3 answering path after the model has produced a
draft answer. It never looks at `executed_answers` or any other gold label.
Instead, it checks for local inconsistencies between the question, selected
evidence, and the model's own `Target:`, `Values:`, `Operation:`,
`Final answer:`, and `Calculation:` lines. When a rule fires, the caller can
retry once with the returned reason.
"""

from __future__ import annotations

import ast
import math
import operator
import re
from collections.abc import Callable, Mapping, Sequence
from typing import cast

from pydantic import BaseModel, ConfigDict

from src.evidence import EvidenceSnippet

_FINAL_ANSWER_PATTERN = re.compile(r"final\s+answer\s*:\s*(?P<value>[^\n\r]+)", re.IGNORECASE)
_LINE_PATTERN_TEMPLATE = r"^\s*{label}\s*:\s*(?P<value>.+?)\s*$"
_NUMBER_PATTERN = re.compile(r"[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)")
_REFUSAL_PATTERN = re.compile(
    r"\b(?:does not provide|not enough information|cannot determine|can't determine|unable to determine)\b",
    re.IGNORECASE,
)
_PERCENT_QUESTION_PATTERN = re.compile(r"\b(?:percent|percentage|portion|ratio|rate)\b", re.IGNORECASE)
_PORTION_OF_TOTAL_PATTERN = re.compile(
    r"\b(?:portion|percentage|percent|ratio)\b.*\b(?:of|related to|from)\b.*\btotal\b"
    r"|\btotal\b.*\b(?:portion|percentage|percent|ratio)\b",
    re.IGNORECASE,
)
_TOTAL_WORD_PATTERN = re.compile(r"\b(?:total|aggregate|overall)\b", re.IGNORECASE)
_ARITHMETIC_CHARS_PATTERN = re.compile(r"[^0-9+\-*/()., ]")

_ALLOWED_OPERATORS: Mapping[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_ALLOWED_UNARY_OPERATORS: Mapping[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


class VerificationResult(BaseModel):
    """Result of checking one draft answer before accepting it."""

    model_config = ConfigDict(extra="forbid")

    should_retry: bool
    reason: str | None = None


class LabeledValue(BaseModel):
    """One value listed by the model on its `Values:` line."""

    model_config = ConfigDict(extra="forbid")

    label: str
    value: float


def verify_answer(
    *,
    question: str,
    answer: str,
    evidence_snippets: Sequence[EvidenceSnippet],
) -> VerificationResult:
    """Return a retry reason when the draft answer is locally suspicious."""
    final_answer = _extract_final_answer(answer)
    if final_answer is None:
        return _retry("The answer did not include a clean `Final answer:` line.")

    clean_final_reason = _find_unclean_final_answer_issue(final_answer)
    if clean_final_reason is not None:
        return _retry(clean_final_reason)

    if _is_refusal(answer):
        if _evidence_has_number(evidence_snippets):
            return _retry(
                "This ConvFinQA question is expected to have an answer in the selected record, "
                "and the selected evidence contains numeric candidates.",
            )
        return _retry("This ConvFinQA question is expected to have an answer in the selected record.")

    if _has_percent_scale_mismatch(question=question, answer=answer, final_answer=final_answer):
        return _retry(
            "The calculation appears to produce a percentage, but the final answer omits the percent sign.",
        )

    competing_zero_reason = _find_competing_zero_selection_issue(
        question=question,
        answer=answer,
        final_answer=final_answer,
    )
    if competing_zero_reason is not None:
        return _retry(competing_zero_reason)

    denominator_reason = _find_denominator_issue(question=question, answer=answer)
    if denominator_reason is not None:
        return _retry(denominator_reason)

    mismatch_reason = _find_calculation_mismatch(answer=answer, final_answer=final_answer)
    if mismatch_reason is not None:
        return _retry(mismatch_reason)

    return VerificationResult(should_retry=False)


def build_retry_instruction(reason: str, question: str) -> str:
    """Build the user message appended for the one v3 correction retry."""
    return "\n".join(
        [
            "Your previous answer may have a calculation or formatting issue.",
            f"Issue: {reason}",
            "",
            "Current question to answer:",
            question,
            "",
            "Re-answer the same question using the selected record and relevant evidence.",
            "Do not answer a previous turn unless the current question explicitly asks for that previous value.",
            "Do not use outside information.",
            "The `Final answer:` line must contain only one comparable value, such as `150` or `3.1%`; do not include words, units, dates, or explanation on that line.",
            "Pay special attention to:",
            "- choosing part / total for portion-of-total questions unless the question clearly says otherwise",
            "- not selecting a zero-valued candidate when another listed non-zero candidate better matches the question target",
            "- percentage scale and percent signs",
            "- operation direction, denominator choice, and final-answer consistency",
            "- returning exactly one clean `Final answer: <value>` line",
        ],
    )


def _retry(reason: str) -> VerificationResult:
    return VerificationResult(should_retry=True, reason=reason)


def _extract_final_answer(answer: str) -> str | None:
    match = _FINAL_ANSWER_PATTERN.search(answer)
    if match is None:
        return None
    return match.group("value").strip()


def _is_refusal(answer: str) -> bool:
    return bool(_REFUSAL_PATTERN.search(answer))


def _find_unclean_final_answer_issue(final_answer: str) -> str | None:
    numbers = _NUMBER_PATTERN.findall(final_answer)
    if len(numbers) > 1:
        return (
            "The `Final answer:` line contains multiple numbers or extra context. "
            "It should contain only one comparable value, with no dates, units, or explanation."
        )

    if numbers and re.search(r"[a-zA-Z]", final_answer.replace("%", "")):
        return (
            "The `Final answer:` line contains extra words. "
            "It should contain only the requested value, such as `150` or `3.1%`."
        )

    return None


def _evidence_has_number(evidence_snippets: Sequence[EvidenceSnippet]) -> bool:
    return any(_NUMBER_PATTERN.search(snippet.text) for snippet in evidence_snippets)


def _has_percent_scale_mismatch(*, question: str, answer: str, final_answer: str) -> bool:
    if not _PERCENT_QUESTION_PATTERN.search(question):
        return False
    if "%" in final_answer:
        return False
    calculation = _line_value(answer, "calculation") or ""
    operation = _line_value(answer, "operation") or ""
    return "%" in calculation or "* 100" in calculation or "*100" in calculation or "* 100" in operation or "*100" in operation


def _find_competing_zero_selection_issue(*, question: str, answer: str, final_answer: str) -> str | None:
    final_value = _parse_number(final_answer)
    if final_value is None or not math.isclose(final_value, 0.0, abs_tol=1e-9):
        return None

    operation = (_line_value(answer, "operation") or "").lower()
    if not operation.startswith("select"):
        return None

    values = _labeled_values(answer)
    selected_label = _selected_label_from_operation(operation)
    if selected_label is None:
        return None

    selected_value = _best_matching_value(selected_label, values)
    if selected_value is None or not math.isclose(selected_value.value, 0.0, abs_tol=1e-9):
        return None

    non_zero_candidates = [value for value in values if not math.isclose(value.value, 0.0, abs_tol=1e-9)]
    if not non_zero_candidates:
        return None

    question_text = question.lower()
    selected_label_tokens = _label_tokens(selected_value.label)
    if selected_label_tokens and selected_label_tokens.issubset(_label_tokens(question_text)):
        return None

    return (
        "The answer selected a zero-valued candidate even though the `Values:` line lists non-zero alternatives. "
        "Re-answer by choosing the value whose label best matches the question target, unless the question explicitly asks for the zero-valued label."
    )


def _find_denominator_issue(*, question: str, answer: str) -> str | None:
    if not _PORTION_OF_TOTAL_PATTERN.search(question):
        return None

    operation = _line_value(answer, "operation") or ""
    numerator, denominator = _extract_division_operands(operation)
    if numerator is None or denominator is None:
        return None

    values_by_number = _values_by_number(answer)
    numerator_label = values_by_number.get(numerator)
    denominator_label = values_by_number.get(denominator)
    if numerator_label is None or denominator_label is None:
        return None

    if _TOTAL_WORD_PATTERN.search(numerator_label) and not _TOTAL_WORD_PATTERN.search(denominator_label):
        return (
            "The question asks for a portion of the total. Your operation appears to divide total by part. "
            "Re-answer using part / total unless the question clearly says otherwise."
        )

    return None


def _find_calculation_mismatch(*, answer: str, final_answer: str) -> str | None:
    final_value = _parse_number(final_answer)
    if final_value is None:
        return None

    expression = _extract_arithmetic_expression(_line_value(answer, "operation") or "")
    if expression is None:
        expression = _extract_arithmetic_expression(_line_value(answer, "calculation") or "")
    if expression is None:
        return None

    calculated_value = _safe_eval_expression(expression)
    if calculated_value is None:
        return None

    expression_is_percent = "*100" in expression.replace(" ", "") or "%" in answer
    comparable_final = final_value
    if "%" in final_answer and not expression_is_percent:
        comparable_final = final_value / 100

    abs_tol = 0.1 if expression_is_percent or "%" in final_answer else 1e-2
    rel_tol = 2e-2 if expression_is_percent or "%" in final_answer else 1e-3
    if math.isclose(calculated_value, comparable_final, rel_tol=rel_tol, abs_tol=abs_tol):
        return None

    return (
        f"The stated operation evaluates to about {calculated_value:g}, "
        f"but the final answer is {final_value:g}."
    )


def _line_value(answer: str, label: str) -> str | None:
    pattern = re.compile(_LINE_PATTERN_TEMPLATE.format(label=re.escape(label)), re.IGNORECASE | re.MULTILINE)
    match = pattern.search(answer)
    if match is None:
        return None
    return match.group("value").strip()


def _values_by_number(answer: str) -> dict[float, str]:
    return {value.value: value.label.lower() for value in _labeled_values(answer)}


def _labeled_values(answer: str) -> list[LabeledValue]:
    values_line = _line_value(answer, "values") or ""
    values: list[LabeledValue] = []
    for part in re.split(r"[,;]", values_line):
        if "=" not in part:
            continue
        label, raw_value = part.rsplit("=", 1)
        value = _parse_number(raw_value)
        if value is not None:
            values.append(LabeledValue(label=label.strip().lower(), value=value))
    return values


def _selected_label_from_operation(operation: str) -> str | None:
    match = re.search(r"\bselect\b\s+(?P<label>.+)", operation, re.IGNORECASE)
    if match is None:
        return None
    return match.group("label").strip().lower()


def _best_matching_value(label: str, values: Sequence[LabeledValue]) -> LabeledValue | None:
    label_tokens = _label_tokens(label)
    best_value: LabeledValue | None = None
    best_overlap = 0
    for value in values:
        overlap = len(label_tokens & _label_tokens(value.label))
        if overlap > best_overlap:
            best_overlap = overlap
            best_value = value
    return best_value


def _label_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z][a-zA-Z0-9]+", text.lower()))


def _extract_division_operands(operation: str) -> tuple[float | None, float | None]:
    cleaned = operation.lower().replace("select", " ")
    match = re.search(
        r"(?P<num>[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+))\s*/\s*"
        r"(?P<den>[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+))",
        cleaned,
    )
    if match is None:
        return None, None
    return _parse_number(match.group("num")), _parse_number(match.group("den"))


def _parse_number(text: str) -> float | None:
    match = _NUMBER_PATTERN.search(text)
    if match is None:
        return None
    return float(match.group().replace(",", ""))


def _extract_arithmetic_expression(text: str) -> str | None:
    without_select = re.sub(r"\bselect\b", " ", text, flags=re.IGNORECASE)
    before_equals = without_select.split("=", 1)[0]
    cleaned = _ARITHMETIC_CHARS_PATTERN.sub(" ", before_equals).replace(",", "")
    # Find the longest arithmetic-looking span containing at least one operator.
    candidates = re.findall(r"[-+*/(). 0-9]+", cleaned)
    candidates = [candidate.strip() for candidate in candidates if _has_binary_operator(candidate)]
    if not candidates:
        return None
    return cast(str, max(candidates, key=len))


def _has_binary_operator(expression: str) -> bool:
    return bool(re.search(r"\d\s*[+\-*/]\s*[-+]?\d", expression))


def _safe_eval_expression(expression: str) -> float | None:
    try:
        node = ast.parse(expression, mode="eval")
        return float(_eval_node(node.body))
    except (SyntaxError, ValueError, ZeroDivisionError, TypeError):
        return None


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)

    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        binary_operation = _ALLOWED_OPERATORS[type(node.op)]
        return float(binary_operation(_eval_node(node.left), _eval_node(node.right)))

    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY_OPERATORS:
        unary_operation = _ALLOWED_UNARY_OPERATORS[type(node.op)]
        return float(unary_operation(_eval_node(node.operand)))

    msg = f"unsupported expression node: {type(node).__name__}"
    raise ValueError(msg)
