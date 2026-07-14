"""Tests for the limited v5 offline numeric fallback."""

# ruff: noqa: D102

from __future__ import annotations

import unittest

from src.evidence import EvidenceSnippet
from src.offline_fallback import answer_with_offline_fallback
from src.prompts import ChatTurn


class OfflineFallbackTests(unittest.TestCase):
    def test_answers_drawn_amount_from_evidence_without_llm(self) -> None:
        result = answer_with_offline_fallback(
            question="as of december 31, 2016, what was the drawn amount from the credit facility that was set to expire in august 2021?",
            history=[],
            evidence_snippets=_entergy_snippets(),
        )

        self.assertTrue(result.answered)
        self.assertGreaterEqual(result.confidence, 0.45)
        self.assertIn("Final answer: 4.7", result.text)
        self.assertIn('"op":"select"', result.text)
        self.assertIn('"evidence":"T-33"', result.text)

    def test_answers_credit_facility_amount_without_llm(self) -> None:
        result = answer_with_offline_fallback(
            question="and what was that credit facility?",
            history=[],
            evidence_snippets=_entergy_snippets(),
        )

        self.assertTrue(result.answered)
        self.assertIn("Final answer: 150", result.text)
        self.assertIn('"evidence":"T-31"', result.text)

    def test_answers_maximum_letter_of_credit_capacity_without_llm(self) -> None:
        result = answer_with_offline_fallback(
            question="and under this same credit facility, what was the maximum amount of letters of credit that could be issued, in millions?",
            history=[
                ChatTurn(user="what was the drawn amount?", assistant="Final answer: 4.7"),
                ChatTurn(user="what was that credit facility?", assistant="Final answer: 150"),
            ],
            evidence_snippets=_entergy_snippets(),
        )

        self.assertTrue(result.answered)
        self.assertIn("Final answer: 75", result.text)
        self.assertIn('"op":"multiply"', result.text)

    def test_abstains_when_candidates_are_not_confident(self) -> None:
        result = answer_with_offline_fallback(
            question="what percentage, then, did that amount represent?",
            history=[
                ChatTurn(user="what was the drawn amount?", assistant="Final answer: 4.7"),
                ChatTurn(user="what was that credit facility?", assistant="Final answer: 150"),
            ],
            evidence_snippets=_entergy_snippets(),
        )

        self.assertFalse(result.answered)
        self.assertIn("unable to determine with offline fallback", result.text)
        self.assertLess(result.confidence, 0.45)


def _entergy_snippets() -> list[EvidenceSnippet]:
    return [
        EvidenceSnippet(
            snippet_id="T-33",
            source="post_text",
            kind="text_sentence",
            text=(
                "as of december 31, 2016, there were no cash borrowings and "
                "$4.7 million of letters of credit outstanding under the credit facility."
            ),
        ),
        EvidenceSnippet(
            snippet_id="T-31",
            source="post_text",
            kind="text_sentence",
            text="entergy texas has a credit facility in the amount of $150 million scheduled to expire in august 2021.",
        ),
        EvidenceSnippet(
            snippet_id="T-32",
            source="post_text",
            kind="text_sentence",
            text="the credit facility allows entergy texas to issue letters of credit against 50% of the borrowing capacity of the facility.",
        ),
    ]


if __name__ == "__main__":
    unittest.main()
