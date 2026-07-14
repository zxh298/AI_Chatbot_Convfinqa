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

    def test_retries_final_answer_with_multiple_numbers(self) -> None:
        result = verify_answer(
            question="and what was that credit facility?",
            answer="Final answer: a credit facility with an amount of 150 million scheduled to expire in August 2021.",
            evidence_snippets=[_snippet("credit facility in the amount of $150 million scheduled to expire in August 2021")],
        )

        self.assertTrue(result.should_retry)
        self.assertIn("multiple numbers", result.reason or "")

    def test_retries_final_answer_with_extra_words(self) -> None:
        result = verify_answer(
            question="and what was that credit facility?",
            answer="Final answer: 150 million",
            evidence_snippets=[_snippet("credit facility in the amount of $150 million")],
        )

        self.assertTrue(result.should_retry)
        self.assertIn("extra words", result.reason or "")

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

    def test_accepts_rounded_percent_final_answer(self) -> None:
        result = verify_answer(
            question="what percentage, then, did that amount represent?",
            answer=(
                "Values: letters of credit outstanding = 4.7, credit facility amount = 150\n"
                "Operation: (4.7 / 150) * 100\n"
                "Final answer: 3.1%\n"
                "Calculation: (4.7 / 150) * 100 = 3.1333333"
            ),
            evidence_snippets=[_snippet("letters of credit outstanding = 4.7 and credit facility = 150")],
        )

        self.assertFalse(result.should_retry)

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

    def test_retries_zero_selection_when_non_zero_candidates_are_listed(self) -> None:
        result = verify_answer(
            question="what was the drawn amount from the credit facility that was set to expire in august 2021?",
            answer=(
                "Target: drawn amount from the credit facility\n"
                "Values: cash borrowings = 0, letters of credit outstanding = 4.7\n"
                "Operation: select cash borrowings\n"
                "Final answer: 0"
            ),
            evidence_snippets=[
                _snippet("there were no cash borrowings and $4.7 million of letters of credit outstanding under the credit facility"),
            ],
        )

        self.assertTrue(result.should_retry)
        self.assertIn("zero-valued candidate", result.reason or "")

    def test_retries_zero_selection_when_non_zero_candidate_only_appears_in_evidence(self) -> None:
        result = verify_answer(
            question="what was the drawn amount from the credit facility that was set to expire in august 2021?",
            answer=(
                "Target: drawn amount from the credit facility\n"
                "Values: cash borrowings = 0\n"
                "Operation: select cash borrowings\n"
                "Final answer: 0"
            ),
            evidence_snippets=[
                _snippet(
                    "as of december 31, 2016, there were no cash borrowings and "
                    "$4.7 million of letters of credit outstanding under the credit facility",
                ),
            ],
        )

        self.assertTrue(result.should_retry)
        self.assertIn("evidence sentence also contains a non-zero numeric alternative", result.reason or "")

    def test_accepts_zero_selection_when_question_asks_for_selected_zero_label(self) -> None:
        result = verify_answer(
            question="what was the amount of cash borrowings?",
            answer=(
                "Target: amount of cash borrowings\n"
                "Values: cash borrowings = 0, letters of credit outstanding = 4.7\n"
                "Operation: select cash borrowings\n"
                "Final answer: 0"
            ),
            evidence_snippets=[
                _snippet("there were no cash borrowings and $4.7 million of letters of credit outstanding under the credit facility"),
            ],
        )

        self.assertFalse(result.should_retry)

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
        instruction = build_retry_instruction(
            reason="The question asks for part / total.",
            question="what was that credit facility?",
        )

        self.assertIn("part / total", instruction)
        self.assertIn("what was that credit facility?", instruction)
        self.assertIn("Do not answer a previous turn", instruction)
        self.assertIn("must contain only one comparable value", instruction)
        self.assertIn("Final answer", instruction)
        self.assertIn("do not change the answer into a refusal", instruction)
        self.assertIn("Calculation plan JSON", instruction)
        self.assertIn("one-step `select` calculation plan", instruction)


def _snippet(text: str) -> EvidenceSnippet:
    return EvidenceSnippet(snippet_id="T-1", source="post_text", kind="text_sentence", text=text)


if __name__ == "__main__":
    unittest.main()
