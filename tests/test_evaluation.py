"""Tests for baseline evaluation helpers."""

# ruff: noqa: D102

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.data import load_dataset
from src.evaluation import build_table4_breakdown, evaluate_records, load_results_jsonl, select_records, write_results_jsonl
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
        self.assertEqual(summary.results[0].prediction_value, 206588.0)
        self.assertEqual(summary.results[0].gold_value, 206588.0)

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

    def test_evaluate_records_keeps_context_isolated_per_record(self) -> None:
        dataset = load_dataset()
        records = dataset.train[:2]
        seen_calls: list[tuple[str, int]] = []

        def answer_fn(
            record: ConvFinQARecord,
            history: list[ChatTurn],
            _question: str,
        ) -> str:
            seen_calls.append((record.id, len(history)))
            return record.dialogue.conv_answers[0]

        evaluate_records(
            records=records,
            answer_fn=answer_fn,
            max_records=2,
            max_turns_per_record=1,
        )

        self.assertEqual(seen_calls, [(records[0].id, 0), (records[1].id, 0)])

    def test_select_records_defaults_to_file_order(self) -> None:
        dataset = load_dataset()

        selected = select_records(dataset.train, max_records=3)

        self.assertEqual([record.id for record in selected], [record.id for record in dataset.train[:3]])

    def test_select_records_random_seed_is_reproducible(self) -> None:
        dataset = load_dataset()

        first_sample = select_records(dataset.train, max_records=5, random_seed=42)
        second_sample = select_records(dataset.train, max_records=5, random_seed=42)

        self.assertEqual([record.id for record in first_sample], [record.id for record in second_sample])
        self.assertNotEqual([record.id for record in first_sample], [record.id for record in dataset.train[:5]])

    def test_write_results_jsonl_saves_raw_and_normalized_answers(self) -> None:
        dataset = load_dataset()
        record = dataset.train[0]

        def answer_fn(
            _record: ConvFinQARecord,
            _history: list[ChatTurn],
            _question: str,
        ) -> str:
            return "Final answer: 14.1%\nCalculation: (206588 - 181001) / 181001"

        summary = evaluate_records(
            records=[record],
            answer_fn=answer_fn,
            max_records=1,
            max_turns_per_record=1,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "eval.jsonl"
            write_results_jsonl(summary, output_path)

            saved_rows = [json.loads(line) for line in output_path.read_text().splitlines()]

        self.assertEqual(len(saved_rows), 1)
        self.assertIn("prediction", saved_rows[0])
        self.assertIn("prediction_value", saved_rows[0])
        self.assertIn("gold_conv_answer", saved_rows[0])
        self.assertIn("gold_value", saved_rows[0])

    def test_load_results_jsonl_round_trips_saved_results(self) -> None:
        dataset = load_dataset()
        record = dataset.train[0]

        def answer_fn(
            _record: ConvFinQARecord,
            _history: list[ChatTurn],
            _question: str,
        ) -> str:
            return record.dialogue.conv_answers[0]

        summary = evaluate_records(records=[record], answer_fn=answer_fn, max_records=1, max_turns_per_record=1)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "eval.jsonl"
            write_results_jsonl(summary, output_path)
            loaded_results = load_results_jsonl(output_path)

        self.assertEqual(len(loaded_results), 1)
        self.assertEqual(loaded_results[0].record_id, record.id)

    def test_build_table4_breakdown_uses_dataset_metadata(self) -> None:
        dataset = load_dataset()
        simple_record = dataset.train[0]
        hybrid_record = next(record for record in dataset.train if record.features.has_type2_question)

        def answer_fn(
            record: ConvFinQARecord,
            _history: list[ChatTurn],
            _question: str,
        ) -> str:
            return record.dialogue.conv_answers[0]

        summary = evaluate_records(
            records=[simple_record, hybrid_record],
            answer_fn=answer_fn,
            max_records=2,
            max_turns_per_record=2,
        )

        rows = {row.label: row for row in build_table4_breakdown(summary.results, dataset)}

        self.assertEqual(rows["full results"].total_turns, 4)
        self.assertGreater(rows["Number selection questions"].total_turns, 0)
        self.assertGreater(rows["Program questions"].total_turns, 0)
        self.assertEqual(rows["Simple conversations"].total_turns, 2)
        self.assertEqual(rows["Hybrid conversations"].total_turns, 2)
        self.assertIn("Turn 0", rows)


if __name__ == "__main__":
    unittest.main()
