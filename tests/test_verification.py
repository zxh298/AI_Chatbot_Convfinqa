"""Tests for v3 no-gold answer verification."""

# ruff: noqa: D102

from __future__ import annotations

import unittest

from src.evidence import EvidenceSnippet
from src.verification import build_retry_instruction, verify_answer


class AnswerVerificationTests(unittest.TestCase):
    def test_accepts_clean_consistent_answer(self) -> None:
        result = verify_answer(
            question="what portion of total is related to performance guarantees?",
            answer=(
                "Values: performance guarantees = 16.3, total = 304.9\n"
                "Operation: (16.3 / 304.9) * 100\n"
                "Calculation: (16.3 / 304.9) * 100 = 5.34%\n"
                "Final answer: 5.34%"
            ),
            evidence_snippets=[_snippet("performance guarantees = 16.3 and total = 304.9")],
        )

        self.assertFalse(result.should_retry)

    def test_retries_missing_final_answer_line(self) -> None:
        result = verify_answer(
            question="what was the credit facility?",
            answer="The credit facility was $150 million.",
            evidence_snippets=[_snippet("credit facility in the amount of $150 million")],
        )

        self.assertTrue(result.should_retry)
        self.assertIn("Final answer", result.reason or "")

    def test_retries_refusal_for_answerable_benchmark_turn(self) -> None:
        result = verify_answer(
            question="what was the credit facility?",
            answer="Final answer: The record does not provide enough information.",
            evidence_snippets=[_snippet("credit facility in the amount of $150 million")],
        )

        self.assertTrue(result.should_retry)
        self.assertIn("expected to have an answer", result.reason or "")

    def test_retries_missing_percent_sign_when_calculation_is_percent(self) -> None:
        result = verify_answer(
            question="what portion is related to performance guarantees?",
            answer=(
                "Values: performance guarantees = 16.3, total = 304.9\n"
                "Operation: (16.3 / 304.9) * 100\n"
                "Calculation: (16.3 / 304.9) * 100 = 5.34%\n"
                "Final answer: 5.34"
            ),
            evidence_snippets=[_snippet("performance guarantees = 16.3 and total = 304.9")],
        )

        self.assertTrue(result.should_retry)
        self.assertIn("percent sign", result.reason or "")

    def test_retries_total_divided_by_part_for_portion_question(self) -> None:
        result = verify_answer(
            question="what portion of total maximum potential amount is related to financial standby letters of credit?",
            answer=(
                "Values: total maximum potential amount = 304.9, financial standby letters of credit = 94.2\n"
                "Operation: 304.9 / 94.2\n"
                "Final answer: 3.24"
            ),
            evidence_snippets=[_snippet("financial standby letters of credit = 94.2 and total = 304.9")],
        )

        self.assertTrue(result.should_retry)
        self.assertIn("divide total by part", result.reason or "")

    def test_retries_final_answer_that_disagrees_with_operation(self) -> None:
        result = verify_answer(
            question="what was the difference?",
            answer=(
                "Values: first value = 245, second value = 167\n"
                "Operation: 245 - 167\n"
                "Final answer: -78"
            ),
            evidence_snippets=[],
        )

        self.assertTrue(result.should_retry)
        self.assertIn("evaluates to about 78", result.reason or "")

    def test_retry_instruction_includes_reason_and_clean_final_answer_rule(self) -> None:
        instruction = build_retry_instruction("The question asks for part / total.")

        self.assertIn("part / total", instruction)
        self.assertIn("Final answer", instruction)


def _snippet(text: str) -> EvidenceSnippet:
    return EvidenceSnippet(snippet_id="T-1", source="post_text", kind="text_sentence", text=text)


if __name__ == "__main__":
    unittest.main()
