"""Tests for record-local evidence snippet selection."""

# ruff: noqa: D102

from __future__ import annotations

import unittest

from src.evidence import (
    build_evidence_snippets,
    build_rerank_messages,
    format_evidence_snippets,
    select_candidate_snippets,
    select_reranked_snippets,
)
from src.models import ConvFinQARecord, Dialogue, Document, Features
from src.prompts import ChatTurn


class EvidenceSelectionTests(unittest.TestCase):
    def test_build_evidence_snippets_uses_text_sentences_and_table_rows(self) -> None:
        record = _make_record()

        snippets = build_evidence_snippets(record)
        snippet_text = "\n".join(snippet.text for snippet in snippets)

        self.assertIn("$4.7 million of letters of credit outstanding", snippet_text)
        self.assertIn("total shares purchased", snippet_text)
        self.assertIn("October: 2506", snippet_text)
        self.assertTrue(any(snippet.kind == "text_sentence" for snippet in snippets))
        self.assertTrue(any(snippet.kind == "table_row" for snippet in snippets))

    def test_long_sentences_also_create_number_centered_clause_snippets(self) -> None:
        record = _make_record()

        snippets = build_evidence_snippets(record)
        number_clauses = [snippet.text for snippet in snippets if snippet.kind == "number_clause"]

        self.assertTrue(any("$150 million of total revolving credit commitments" in text for text in number_clauses))

    def test_candidate_filtering_uses_current_question_and_history(self) -> None:
        record = _make_record()
        snippets = build_evidence_snippets(record)
        history = [
            ChatTurn(
                user="what was the outstanding amount of letters of credit?",
                assistant="Final answer: 4.7",
            ),
        ]

        candidates = select_candidate_snippets(
            snippets=snippets,
            history=history,
            current_question="what percentage of the facility was used?",
            limit=5,
        )
        candidate_text = "\n".join(snippet.text for snippet in candidates)

        self.assertIn("$4.7 million of letters of credit outstanding", candidate_text)
        self.assertIn("$150 million", candidate_text)

    def test_build_rerank_messages_includes_history_question_and_candidates(self) -> None:
        record = _make_record()
        candidates = build_evidence_snippets(record)[:2]
        history = [ChatTurn(user="previous question?", assistant="Final answer: 10")]

        messages = build_rerank_messages(
            history=history,
            current_question="current question?",
            candidate_snippets=candidates,
        )

        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("previous question?", messages[1]["content"])
        self.assertIn("current question?", messages[1]["content"])
        self.assertIn(candidates[0].snippet_id, messages[1]["content"])

    def test_select_reranked_snippets_parses_ids_and_falls_back_to_candidate_order(self) -> None:
        record = _make_record()
        candidates = build_evidence_snippets(record)[:3]
        response = f"{candidates[2].snippet_id}, {candidates[0].snippet_id}"

        selected = select_reranked_snippets(candidates, response, final_limit=3)

        self.assertEqual([snippet.snippet_id for snippet in selected[:2]], [candidates[2].snippet_id, candidates[0].snippet_id])
        self.assertEqual(len(selected), 3)

    def test_format_evidence_snippets_is_prompt_readable(self) -> None:
        record = _make_record()
        snippet = build_evidence_snippets(record)[0]

        formatted = format_evidence_snippets([snippet])

        self.assertIn("Relevant evidence:", formatted)
        self.assertIn(snippet.snippet_id, formatted)
        self.assertIn(snippet.text, formatted)


def _make_record() -> ConvFinQARecord:
    return ConvFinQARecord(
        id="demo-record",
        doc=Document(
            pre_text="The company reported share repurchase activity during the quarter.",
            table={
                "October": {"total shares purchased": 2506},
                "November": {"total shares purchased": 1923},
                "December": {"total shares purchased": 1379},
            },
            post_text=(
                "For 2016, the company had no cash borrowings outstanding under the credit facility, "
                "$4.7 million of letters of credit outstanding, and $150 million of total revolving "
                "credit commitments. The company also discussed liquidity after year end."
            ),
        ),
        dialogue=Dialogue(
            conv_questions=["question"],
            conv_answers=["answer"],
            turn_program=["0"],
            executed_answers=[0],
            qa_split=[False],
        ),
        features=Features(
            num_dialogue_turns=1,
            has_type2_question=False,
            has_duplicate_columns=False,
            has_non_numeric_values=False,
        ),
    )


if __name__ == "__main__":
    unittest.main()
