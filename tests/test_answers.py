"""Tests for answer normalization and comparison."""

# ruff: noqa: D102

from __future__ import annotations

import unittest

from src.answers import (
    answers_match,
    display_answers_match,
    normalize_gold_answer,
    parse_answer_text,
)


class AnswerNormalizationTests(unittest.TestCase):
    def test_percent_text_is_normalized_to_ratio(self) -> None:
        parsed = parse_answer_text("-3.3%")

        self.assertTrue(parsed.is_numeric)
        self.assertTrue(parsed.is_percent)
        self.assertAlmostEqual(parsed.value, -0.033)

    def test_negative_word_percent_is_normalized_to_ratio(self) -> None:
        parsed = parse_answer_text("negative 3.3%")

        self.assertTrue(parsed.is_numeric)
        self.assertTrue(parsed.is_percent)
        self.assertAlmostEqual(parsed.value, -0.033)

    def test_gold_percent_prefers_display_answer_scale(self) -> None:
        normalized = normalize_gold_answer(-0.03264, "-3.3%")

        self.assertTrue(normalized.is_numeric)
        self.assertTrue(normalized.is_percent)
        self.assertAlmostEqual(normalized.value, -0.033)

    def test_percent_prediction_matches_executed_ratio_gold(self) -> None:
        self.assertTrue(answers_match("negative 3.3%", -0.03264))

    def test_final_answer_line_is_preferred_over_calculation(self) -> None:
        prediction = "Calculation: (326 - 337) / 337 = -0.03264\nFinal answer: -3.3%"

        self.assertTrue(answers_match(prediction, -0.03264))

    def test_strict_final_answer_value_ignores_explanation_line(self) -> None:
        prediction = "Final answer: 3\nCalculation: 11 - 8 = 3 million."

        self.assertTrue(answers_match(prediction, 3.0))

    def test_fallback_uses_last_number_when_no_final_answer_line(self) -> None:
        prediction = "Calculation: (326 - 337) / 337 = -0.03264, or -3.3%."

        self.assertTrue(answers_match(prediction, -0.03264))

    def test_plain_number_prediction_matches_plain_gold(self) -> None:
        self.assertTrue(answers_match("25,587", 25587.0))

    def test_leading_decimal_gold_answer_is_parsed_as_decimal(self) -> None:
        normalized = normalize_gold_answer(0.0751, ".0751")

        self.assertTrue(normalized.is_numeric)
        self.assertAlmostEqual(normalized.value, 0.0751)

    def test_negative_leading_decimal_prediction_is_parsed_as_decimal(self) -> None:
        parsed = parse_answer_text("Final answer: -.0751")

        self.assertTrue(parsed.is_numeric)
        self.assertAlmostEqual(parsed.value, -0.0751)

    def test_wrong_percent_prediction_does_not_match(self) -> None:
        self.assertFalse(answers_match("-4.3%", -0.03264))

    def test_display_normalized_scoring_can_be_stricter_than_strict_execution(self) -> None:
        prediction = "Final answer: 2.5%\nCalculation: (11.9 / 486.9) * 100 = 2.5%"

        self.assertTrue(answers_match(prediction, 0.02444))
        self.assertFalse(display_answers_match(prediction, 0.02444, "2.4%"))

    def test_display_normalized_scoring_can_be_more_forgiving_than_strict_execution(self) -> None:
        prediction = "Final answer: 128"

        self.assertFalse(answers_match(prediction, 2.28))
        self.assertTrue(display_answers_match(prediction, 2.28, "128"))


if __name__ == "__main__":
    unittest.main()
