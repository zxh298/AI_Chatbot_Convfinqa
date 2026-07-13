"""Tests for baseline evaluation helpers."""

# ruff: noqa: D102

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.data import load_dataset
from src.evaluation import (
    append_run_result_jsonl,
    build_table4_breakdown,
    deduplicate_run_results,
    evaluate_run_results,
    load_results_jsonl,
    load_run_jsonl,
    load_selected_record_ids_jsonl,
    run_records,
    run_records_parallel,
    select_records,
    write_results_jsonl,
    write_run_jsonl,
    write_selected_records_jsonl,
)
from src.models import ConvFinQARecord
from src.prompts import ChatTurn


class BaselineEvaluationTests(unittest.TestCase):
    def test_run_records_replays_turns_and_saves_raw_predictions(self) -> None:
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

        summary = run_records(
            records=[record],
            answer_fn=answer_fn,
            max_records=1,
            max_turns_per_record=2,
        )

        self.assertEqual(summary.total_turns, 2)
        self.assertEqual(summary.results[0].prediction, record.dialogue.conv_answers[0])
        self.assertEqual(summary.results[1].prediction, "wrong answer")
        self.assertFalse(hasattr(summary.results[0], "is_correct"))

    def test_run_records_preserves_history_between_turns(self) -> None:
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

        run_records(
            records=[record],
            answer_fn=answer_fn,
            max_records=1,
            max_turns_per_record=3,
        )

        self.assertEqual(seen_history_lengths, [0, 1, 2])

    def test_run_records_keeps_context_isolated_per_record(self) -> None:
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

        run_records(
            records=records,
            answer_fn=answer_fn,
            max_records=2,
            max_turns_per_record=1,
        )

        self.assertEqual(seen_calls, [(records[0].id, 0), (records[1].id, 0)])

    def test_run_records_parallel_preserves_record_order_and_turn_history(self) -> None:
        dataset = load_dataset()
        records = dataset.train[:2]
        seen_history_lengths_by_record: dict[str, list[int]] = {record.id: [] for record in records}

        def answer_fn(
            record: ConvFinQARecord,
            history: list[ChatTurn],
            _question: str,
        ) -> str:
            seen_history_lengths_by_record[record.id].append(len(history))
            return record.dialogue.conv_answers[len(history)]

        summary = run_records_parallel(
            records=records,
            answer_fn=answer_fn,
            max_records=2,
            max_turns_per_record=2,
            workers=2,
        )

        self.assertEqual(summary.total_turns, 4)
        self.assertEqual([result.record_id for result in summary.results], [records[0].id, records[0].id, records[1].id, records[1].id])
        self.assertEqual(seen_history_lengths_by_record[records[0].id], [0, 1])
        self.assertEqual(seen_history_lengths_by_record[records[1].id], [0, 1])

    def test_select_records_defaults_to_file_order(self) -> None:
        dataset = load_dataset()

        selected = select_records(dataset.train, max_records=3)

        self.assertEqual([record.id for record in selected], [record.id for record in dataset.train[:3]])

    def test_select_records_without_limit_returns_all_records(self) -> None:
        dataset = load_dataset()

        selected = select_records(dataset.train, max_records=None)

        self.assertEqual(len(selected), len(dataset.train))
        self.assertEqual([record.id for record in selected], [record.id for record in dataset.train])

    def test_select_records_random_seed_is_reproducible(self) -> None:
        dataset = load_dataset()

        first_sample = select_records(dataset.train, max_records=5, random_seed=42)
        second_sample = select_records(dataset.train, max_records=5, random_seed=42)

        self.assertEqual([record.id for record in first_sample], [record.id for record in second_sample])
        self.assertNotEqual([record.id for record in first_sample], [record.id for record in dataset.train[:5]])

    def test_write_run_jsonl_saves_raw_predictions_only(self) -> None:
        dataset = load_dataset()
        record = dataset.train[0]

        def answer_fn(
            _record: ConvFinQARecord,
            _history: list[ChatTurn],
            _question: str,
        ) -> str:
            return "Final answer: 14.1%\nCalculation: (206588 - 181001) / 181001"

        summary = run_records(
            records=[record],
            answer_fn=answer_fn,
            max_records=1,
            max_turns_per_record=1,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "run.jsonl"
            write_run_jsonl(summary, output_path)

            saved_rows = [json.loads(line) for line in output_path.read_text().splitlines()]

        self.assertEqual(len(saved_rows), 1)
        self.assertIn("prediction", saved_rows[0])
        self.assertNotIn("is_correct", saved_rows[0])
        self.assertNotIn("gold_executed_answer", saved_rows[0])

    def test_append_run_result_jsonl_saves_each_completed_turn(self) -> None:
        dataset = load_dataset()
        record = dataset.train[0]
        first_result = run_records(
            records=[record],
            answer_fn=lambda _record, _history, _question: "Final answer: 1",
            max_records=1,
            max_turns_per_record=1,
        ).results[0]
        second_result = first_result.model_copy(update={"turn_index": 1, "prediction": "Final answer: 2"})

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "run.jsonl"
            append_run_result_jsonl(first_result, output_path)
            append_run_result_jsonl(second_result, output_path)
            loaded_results = load_run_jsonl(output_path)

        self.assertEqual([result.prediction for result in loaded_results], ["Final answer: 1", "Final answer: 2"])

    def test_write_selected_records_jsonl_saves_sample_order(self) -> None:
        dataset = load_dataset()
        records = select_records(dataset.train, max_records=3, random_seed=42)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "records.jsonl"
            write_selected_records_jsonl(records, output_path)
            saved_rows = [json.loads(line) for line in output_path.read_text().splitlines()]
            loaded_record_ids = load_selected_record_ids_jsonl(output_path)

        self.assertEqual([row["index"] for row in saved_rows], [0, 1, 2])
        self.assertEqual([row["record_id"] for row in saved_rows], [record.id for record in records])
        self.assertEqual(loaded_record_ids, [record.id for record in records])

    def test_deduplicate_run_results_keeps_latest_duplicate_turn(self) -> None:
        dataset = load_dataset()
        record = dataset.train[0]
        base_result = run_records(
            records=[record],
            answer_fn=lambda _record, _history, _question: "old",
            max_records=1,
            max_turns_per_record=1,
        ).results[0]
        latest_result = base_result.model_copy(update={"prediction": "new"})

        deduped = deduplicate_run_results([base_result, latest_result])

        self.assertEqual(len(deduped), 1)
        self.assertEqual(deduped[0].prediction, "new")

    def test_evaluate_run_results_uses_strict_executed_gold_for_correctness(self) -> None:
        dataset = load_dataset()
        record = next(record for record in dataset.train if record.id == "Single_STT/2013/page_54.pdf-4")

        def answer_fn(
            _record: ConvFinQARecord,
            _history: list[ChatTurn],
            _question: str,
        ) -> str:
            return "Final answer: 128"

        run_summary = run_records(
            records=[record],
            answer_fn=answer_fn,
            max_records=1,
            max_turns_per_record=1,
        )
        summary = evaluate_run_results(run_summary.results, dataset)

        self.assertEqual(summary.results[0].gold_conv_answer, "128")
        self.assertEqual(summary.results[0].gold_executed_answer, 2.28)
        self.assertEqual(summary.results[0].gold_value, 2.28)
        self.assertFalse(summary.results[0].is_correct)

    def test_load_run_jsonl_round_trips_saved_raw_predictions(self) -> None:
        dataset = load_dataset()
        record = dataset.train[0]

        def answer_fn(
            _record: ConvFinQARecord,
            _history: list[ChatTurn],
            _question: str,
        ) -> str:
            return record.dialogue.conv_answers[0]

        summary = run_records(records=[record], answer_fn=answer_fn, max_records=1, max_turns_per_record=1)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "run.jsonl"
            write_run_jsonl(summary, output_path)
            loaded_results = load_run_jsonl(output_path)

        self.assertEqual(len(loaded_results), 1)
        self.assertEqual(loaded_results[0].record_id, record.id)
        self.assertEqual(loaded_results[0].prediction, record.dialogue.conv_answers[0])

    def test_write_results_jsonl_saves_scored_results(self) -> None:
        dataset = load_dataset()
        record = dataset.train[0]

        def answer_fn(
            _record: ConvFinQARecord,
            _history: list[ChatTurn],
            _question: str,
        ) -> str:
            return record.dialogue.conv_answers[0]

        run_summary = run_records(records=[record], answer_fn=answer_fn, max_records=1, max_turns_per_record=1)
        evaluation_summary = evaluate_run_results(run_summary.results, dataset)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "eval.jsonl"
            write_results_jsonl(evaluation_summary, output_path)
            loaded_results = load_results_jsonl(output_path)

        self.assertEqual(len(loaded_results), 1)
        self.assertIn("is_correct", loaded_results[0].model_dump())
        self.assertIn("gold_executed_answer", loaded_results[0].model_dump())

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

        run_summary = run_records(
            records=[simple_record, hybrid_record],
            answer_fn=answer_fn,
            max_records=2,
            max_turns_per_record=2,
        )
        summary = evaluate_run_results(run_summary.results, dataset)

        rows = {row.label: row for row in build_table4_breakdown(summary.results, dataset)}

        self.assertEqual(rows["full results"].total_turns, 4)
        self.assertGreater(rows["Number selection questions"].total_turns, 0)
        self.assertGreater(rows["Program questions"].total_turns, 0)
        self.assertEqual(rows["Simple conversations"].total_turns, 2)
        self.assertEqual(rows["Hybrid conversations"].total_turns, 2)
        self.assertIn("Turn 0", rows)


if __name__ == "__main__":
    unittest.main()
