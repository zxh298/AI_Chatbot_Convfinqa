"""Lightweight train-example retrieval for reasoning-pattern guidance.

This module supports the v4 answering path. It retrieves similar solved turns
from training records using local lexical scoring only. Retrieved examples are
not evidence for the current answer; they are pattern hints that help the model
choose operations such as part/total, difference direction, or follow-up
reference handling.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from src.models import ConvFinQARecord, TableValue

if TYPE_CHECKING:
    from src.prompts import ChatTurn

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_SINGLE_RECORD_SUFFIX_PATTERN = re.compile(r"-\d+$")
_PATTERN_TERMS = {
    "amount",
    "change",
    "difference",
    "percent",
    "percentage",
    "portion",
    "proportion",
    "ratio",
    "rate",
    "that",
    "this",
    "total",
    "value",
}
_PERCENT_PATTERN_TERMS = {"percent", "percentage", "portion", "proportion", "ratio", "rate"}
_DIFFERENCE_PATTERN_TERMS = {"change", "difference"}
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
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


class ReasoningExample(BaseModel):
    """One solved training turn used as a reasoning-pattern example."""

    model_config = ConfigDict(extra="forbid")

    record_id: str
    turn_index: int
    previous_question: str | None = None
    previous_answer: str | None = None
    question: str
    turn_program: str
    executed_answer: TableValue
    conv_answer: str


class ExampleIndexItem(BaseModel):
    """Indexed text plus the solved example metadata."""

    model_config = ConfigDict(extra="forbid")

    example: ReasoningExample
    retrieval_text: str
    tokens: set[str]
    pattern_tokens: set[str]
    question_pattern_tokens: set[str]


def build_example_index(records: Sequence[ConvFinQARecord]) -> list[ExampleIndexItem]:
    """Build an in-memory lexical index from training records."""
    items: list[ExampleIndexItem] = []
    for record in records:
        turn_count = min(
            len(record.dialogue.conv_questions),
            len(record.dialogue.conv_answers),
            len(record.dialogue.turn_program),
            len(record.dialogue.executed_answers),
        )
        for turn_index in range(turn_count):
            previous_question = record.dialogue.conv_questions[turn_index - 1] if turn_index > 0 else None
            previous_answer = record.dialogue.conv_answers[turn_index - 1] if turn_index > 0 else None
            question = record.dialogue.conv_questions[turn_index]
            question_tokens = _tokens(question)
            retrieval_text = " ".join(
                part
                for part in [
                    previous_question or "",
                    previous_answer or "",
                    question,
                    record.dialogue.turn_program[turn_index],
                ]
                if part
            )
            tokens = _tokens(retrieval_text)
            items.append(
                ExampleIndexItem(
                    example=ReasoningExample(
                        record_id=record.id,
                        turn_index=turn_index,
                        previous_question=previous_question,
                        previous_answer=previous_answer,
                        question=question,
                        turn_program=record.dialogue.turn_program[turn_index],
                        executed_answer=record.dialogue.executed_answers[turn_index],
                        conv_answer=record.dialogue.conv_answers[turn_index],
                    ),
                    retrieval_text=retrieval_text,
                    tokens=tokens,
                    pattern_tokens=tokens & _PATTERN_TERMS,
                    question_pattern_tokens=question_tokens & _PATTERN_TERMS,
                ),
            )
    return items


def retrieve_reasoning_examples(
    index: Sequence[ExampleIndexItem],
    *,
    history: Sequence[ChatTurn],
    current_question: str,
    current_record_id: str,
    limit: int = 3,
) -> list[ReasoningExample]:
    """Return similar train examples while excluding the current record."""
    current_document_key = record_document_key(current_record_id)
    query_text = " ".join(
        [
            *[
                f"{turn.user} {turn.assistant}"
                for turn in history[-2:]
            ],
            current_question,
        ],
    )
    query_tokens = _tokens(query_text)
    query_pattern_tokens = query_tokens & _PATTERN_TERMS
    current_turn_index = len(history)

    scored: list[tuple[int, int, ExampleIndexItem]] = []
    for index_position, item in enumerate(index):
        if record_document_key(item.example.record_id) == current_document_key:
            continue
        if not _passes_pattern_filter(item, query_pattern_tokens):
            continue
        score = _score_example(
            item=item,
            query_tokens=query_tokens,
            query_pattern_tokens=query_pattern_tokens,
            current_turn_index=current_turn_index,
        )
        if score > 0:
            scored.append((score, index_position, item))

    scored.sort(key=lambda value: (-value[0], value[1]))
    return [item.example for _score, _index_position, item in scored[:limit]]


def format_reasoning_examples(examples: Sequence[ReasoningExample]) -> str:
    """Format retrieved examples for the answer prompt."""
    if not examples:
        return "Similar solved train examples:\n(none selected)"

    lines = [
        "Similar solved train examples:",
        "Use these only as reasoning-pattern hints. Do not copy their numbers.",
    ]
    for index, example in enumerate(examples, start=1):
        lines.extend(
            [
                f"Example {index}:",
                f"- Question: {example.question}",
                f"- Previous turn: {_format_previous_turn(example)}",
                f"- Reasoning program: {example.turn_program}",
                f"- Example answer: {example.conv_answer} (executed: {example.executed_answer})",
            ],
        )
    return "\n".join(lines)


def _score_example(
    *,
    item: ExampleIndexItem,
    query_tokens: set[str],
    query_pattern_tokens: set[str],
    current_turn_index: int,
) -> int:
    shared_tokens = item.tokens & query_tokens
    shared_pattern_tokens = item.pattern_tokens & query_pattern_tokens
    score = len(shared_tokens) + 3 * len(shared_pattern_tokens)
    if item.example.turn_index == current_turn_index:
        score += 2
    if _same_question_shape(item.example.question, query_tokens):
        score += 2
    return score


def _passes_pattern_filter(item: ExampleIndexItem, query_pattern_tokens: set[str]) -> bool:
    """Keep examples aligned with strong math-pattern words in the query."""
    if query_pattern_tokens & _PERCENT_PATTERN_TERMS:
        return bool(item.question_pattern_tokens & _PERCENT_PATTERN_TERMS) or "divide(" in item.example.turn_program
    if query_pattern_tokens & _DIFFERENCE_PATTERN_TERMS:
        return bool(item.question_pattern_tokens & _DIFFERENCE_PATTERN_TERMS) or "subtract(" in item.example.turn_program
    return True


def _same_question_shape(question: str, query_tokens: set[str]) -> bool:
    question_tokens = _tokens(question)
    return bool((question_tokens & query_tokens) & _PATTERN_TERMS)


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in _TOKEN_PATTERN.findall(text.lower())
        if len(token) > 1 and token not in _STOPWORDS
    }


def record_document_key(record_id: str) -> str:
    """Normalize Single/Double variants to the underlying PDF page."""
    if "_" in record_id:
        _prefix, record_id = record_id.split("_", 1)
    return _SINGLE_RECORD_SUFFIX_PATTERN.sub("", record_id)


def _format_previous_turn(example: ReasoningExample) -> str:
    if example.previous_question is None or example.previous_answer is None:
        return "(none)"
    return f"Q: {example.previous_question} A: {example.previous_answer}"
