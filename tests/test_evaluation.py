"""Tests for baseline evaluation helpers."""

# ruff: noqa: D102

from __future__ import annotations

import unittest

from src.data import load_dataset
from src.evaluation import evaluate_records
from src.models import ConvFinQARecord
from src.prompts import ChatTurn


class BaselineEvaluationTests(unittest.TestCase):
    def test_evaluate_records_replays_turns_and_tracks_accuracy(self) -> None:
        dataset = load_dataset()
        record = dataset.train[0]

        def answer_fn(
            _record: ConvFinQARecord,
            _history: list[ChatTurn],
            question: str,
        ) -> str:
            if question == record.dialogue.conv_questions[0]:
                return record.dialogue.conv_answers[0]
            return "wrong answer"

        summary = evaluate_records(
            records=[record],
            answer_fn=answer_fn,
            max_records=1,
            max_turns_per_record=2,
        )

        self.assertEqual(summary.total_turns, 2)
        self.assertEqual(summary.correct_turns, 1)
        self.assertAlmostEqual(summary.accuracy, 0.5)
        self.assertTrue(summary.results[0].is_correct)
        self.assertFalse(summary.results[1].is_correct)

    def test_evaluate_records_preserves_history_between_turns(self) -> None:
        dataset = load_dataset()
        record = dataset.train[0]
        seen_history_lengths: list[int] = []

        def answer_fn(
            _record: ConvFinQARecord,
            history: list[ChatTurn],
            _question: str,
        ) -> str:
            seen_history_lengths.append(len(history))
            return "0"

        evaluate_records(
            records=[record],
            answer_fn=answer_fn,
            max_records=1,
            max_turns_per_record=3,
        )

        self.assertEqual(seen_history_lengths, [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
