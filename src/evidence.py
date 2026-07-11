"""Record-local evidence snippets for ConvFinQA prompts.

The document is already selected by ``record_id``, so this module focuses on
finding useful evidence inside that one record rather than searching across the
whole dataset.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from src.models import ConvFinQARecord, TableValue

if TYPE_CHECKING:
    from src.prompts import ChatTurn

_NUMBER_PATTERN = re.compile(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)%?")
_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_YEAR_OR_MONTH_PATTERN = re.compile(
    r"\b(?:19|20)\d{2}\b|\b(?:january|february|march|april|may|june|july|august|"
    r"september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|"
    r"oct|nov|dec)\b",
    re.IGNORECASE,
)
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
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
    "with",
}


class EvidenceSnippet(BaseModel):
    """One readable evidence unit extracted from a selected record."""

    model_config = ConfigDict(extra="forbid")

    snippet_id: str
    source: str
    kind: str
    text: str


def build_evidence_snippets(record: ConvFinQARecord) -> list[EvidenceSnippet]:
    """Build deterministic evidence snippets from text and table context."""
    snippets: list[EvidenceSnippet] = []

    snippets.extend(_build_text_snippets(record.doc.pre_text, source="pre_text", start_index=len(snippets) + 1))
    snippets.extend(_build_table_snippets(record, start_index=len(snippets) + 1))
    snippets.extend(_build_text_snippets(record.doc.post_text, source="post_text", start_index=len(snippets) + 1))

    return _deduplicate_snippets(snippets)


def select_candidate_snippets(
    snippets: Sequence[EvidenceSnippet],
    history: Sequence[ChatTurn],
    current_question: str,
    limit: int = 30,
) -> list[EvidenceSnippet]:
    """Use cheap lexical matching to choose reranker candidates."""
    scored = [
        (_score_snippet(snippet, history=history, current_question=current_question), index, snippet)
        for index, snippet in enumerate(snippets)
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [snippet for score, _index, snippet in scored[:limit] if score > 0] or list(snippets[:limit])


def build_rerank_messages(
    history: Sequence[ChatTurn],
    current_question: str,
    candidate_snippets: Sequence[EvidenceSnippet],
    final_limit: int = 8,
) -> list[dict[str, str]]:
    """Build messages for LLM evidence reranking."""
    history_text = _format_history(history)
    candidates_text = "\n".join(
        f"[{snippet.snippet_id}] {snippet.text}" for snippet in candidate_snippets
    )

    system_prompt = "\n".join(
        [
            "You select evidence snippets for financial question answering.",
            "Use the current question and prior Q/A only to understand references.",
            "Select only from the candidate document snippets.",
            "Prefer snippets that identify exact values, periods, units, numerator, and denominator.",
            "Return only snippet IDs in ranked order, separated by commas.",
            f"Return at most {final_limit} IDs.",
        ],
    )
    user_prompt = "\n\n".join(
        [
            "Conversation history:\n" + (history_text or "(none)"),
            "Current question:\n" + current_question,
            "Candidate snippets:\n" + candidates_text,
        ],
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def select_reranked_snippets(
    candidate_snippets: Sequence[EvidenceSnippet],
    reranker_response: str,
    final_limit: int = 8,
) -> list[EvidenceSnippet]:
    """Parse reranker-selected IDs, falling back to candidate order."""
    snippets_by_id = {snippet.snippet_id: snippet for snippet in candidate_snippets}
    selected: list[EvidenceSnippet] = []

    for snippet_id in re.findall(r"\b[A-Z]+-\d+\b", reranker_response.upper()):
        snippet = snippets_by_id.get(snippet_id)
        if snippet is not None and snippet not in selected:
            selected.append(snippet)
        if len(selected) >= final_limit:
            return selected

    fallback = [snippet for snippet in candidate_snippets if snippet not in selected]
    return [*selected, *fallback][:final_limit]


def format_evidence_snippets(snippets: Sequence[EvidenceSnippet]) -> str:
    """Format selected snippets for the answer prompt."""
    if not snippets:
        return "Relevant evidence:\n(none selected)"
    lines = ["Relevant evidence:"]
    lines.extend(f"- [{snippet.snippet_id}] {snippet.text}" for snippet in snippets)
    return "\n".join(lines)


def _build_text_snippets(text: str, source: str, start_index: int) -> list[EvidenceSnippet]:
    snippets: list[EvidenceSnippet] = []
    next_index = start_index

    for sentence in _split_sentences(text):
        snippets.append(
            EvidenceSnippet(
                snippet_id=f"T-{next_index}",
                source=source,
                kind="text_sentence",
                text=sentence,
            ),
        )
        next_index += 1

        if len(sentence) > 160 and len(_NUMBER_PATTERN.findall(sentence)) > 1:
            for clause in _split_number_clauses(sentence):
                snippets.append(
                    EvidenceSnippet(
                        snippet_id=f"T-{next_index}",
                        source=source,
                        kind="number_clause",
                        text=clause,
                    ),
                )
                next_index += 1

    return snippets


def _build_table_snippets(record: ConvFinQARecord, start_index: int) -> list[EvidenceSnippet]:
    snippets: list[EvidenceSnippet] = []
    columns = list(record.doc.table.keys())

    for row_index, row_label in enumerate(_ordered_row_labels(record.doc.table), start=start_index):
        values = [
            f"{column}: {_format_value(record.doc.table[column].get(row_label, ''))}"
            for column in columns
            if row_label in record.doc.table[column]
        ]
        snippets.append(
            EvidenceSnippet(
                snippet_id=f"R-{row_index}",
                source="table",
                kind="table_row",
                text=f"{row_label} | " + " | ".join(values),
            ),
        )

    return snippets


def _split_sentences(text: str) -> list[str]:
    normalized_text = _normalize_space(text)
    if not normalized_text:
        return []
    return [
        sentence.strip()
        for sentence in _SENTENCE_SPLIT_PATTERN.split(normalized_text)
        if sentence.strip()
    ]


def _split_number_clauses(sentence: str) -> list[str]:
    clauses = [
        _normalize_space(clause)
        for clause in re.split(r"[;,]", sentence)
        if _NUMBER_PATTERN.search(clause)
    ]
    return [clause for clause in clauses if len(clause.split()) >= 4]


def _deduplicate_snippets(snippets: Iterable[EvidenceSnippet]) -> list[EvidenceSnippet]:
    deduplicated: list[EvidenceSnippet] = []
    seen_text: set[str] = set()

    for index, snippet in enumerate(snippets, start=1):
        normalized_text = snippet.text.lower()
        if normalized_text in seen_text:
            continue
        seen_text.add(normalized_text)
        prefix = "R" if snippet.kind == "table_row" else "T"
        deduplicated.append(snippet.model_copy(update={"snippet_id": f"{prefix}-{index}"}))

    return deduplicated


def _score_snippet(
    snippet: EvidenceSnippet,
    history: Sequence[ChatTurn],
    current_question: str,
) -> int:
    snippet_tokens = _tokens(snippet.text)
    question_tokens = _tokens(current_question)
    history_tokens = _tokens(_format_history(history[-2:]))
    date_tokens = _date_tokens(current_question)

    score = 0
    score += 3 * len(snippet_tokens & question_tokens)
    score += len(snippet_tokens & history_tokens)
    score += 3 * len(_date_tokens(snippet.text) & date_tokens)

    if _asks_for_number(current_question) and _NUMBER_PATTERN.search(snippet.text):
        score += 2
    if snippet.kind == "table_row" and snippet_tokens & question_tokens:
        score += 1

    return score


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in _TOKEN_PATTERN.findall(text.lower())
        if len(token) > 1 and token not in _STOPWORDS
    }


def _date_tokens(text: str) -> set[str]:
    return {match.group(0).lower() for match in _YEAR_OR_MONTH_PATTERN.finditer(text)}


def _asks_for_number(question: str) -> bool:
    question_tokens = _tokens(question)
    return bool(
        question_tokens
        & {
            "amount",
            "average",
            "change",
            "difference",
            "fraction",
            "how",
            "many",
            "much",
            "percent",
            "percentage",
            "portion",
            "ratio",
            "total",
        },
    )


def _format_history(history: Sequence[ChatTurn]) -> str:
    return "\n".join(
        f"Q: {turn.user}\nA: {turn.assistant}"
        for turn in history
    )


def _ordered_row_labels(table: dict[str, dict[str, TableValue]]) -> list[str]:
    row_labels: list[str] = []
    seen: set[str] = set()
    for column_values in table.values():
        for row_label in column_values:
            if row_label not in seen:
                seen.add(row_label)
                row_labels.append(row_label)
    return row_labels


def _format_value(value: TableValue | str) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _normalize_space(text: str) -> str:
    return " ".join(text.split())
