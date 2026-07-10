"""Tests for record loading, formatting, and prompt assembly."""

# ruff: noqa: D102

from __future__ import annotations

import unittest

from src.data import find_record, load_dataset
from src.formatting import format_record_context, format_table
from src.prompts import ChatTurn, build_chat_messages, build_system_prompt


class Stage1Tests(unittest.TestCase):
    def test_dataset_loads_and_validates_records(self) -> None:
        dataset = load_dataset()

        self.assertEqual(len(dataset.train), 3037)
        self.assertEqual(len(dataset.dev), 421)
        self.assertEqual(dataset.train[0].id, "Single_JKHY/2009/page_28.pdf-3")

    def test_find_record_returns_split_and_record(self) -> None:
        dataset = load_dataset()

        located = find_record(dataset, "Single_JKHY/2009/page_28.pdf-3")

        self.assertIsNotNone(located)
        assert located is not None
        self.assertEqual(located.split, "train")
        self.assertEqual(located.record.features.num_dialogue_turns, 4)

    def test_find_record_returns_none_for_unknown_id(self) -> None:
        dataset = load_dataset()

        located = find_record(dataset, "missing-record-id")

        self.assertIsNone(located)

    def test_format_table_converts_column_oriented_table_to_markdown(self) -> None:
        table = {
            "2009": {"net income": 10.0, "cash": 5},
            "2008": {"net income": 8.0, "cash": "n/a"},
        }

        markdown = format_table(table)

        self.assertIn("| metric | 2009 | 2008 |", markdown)
        self.assertIn("| net income | 10 | 8 |", markdown)
        self.assertIn("| cash | 5 | n/a |", markdown)

    def test_record_context_contains_text_and_table(self) -> None:
        dataset = load_dataset()
        record = dataset.train[0]

        context = format_record_context(record)

        self.assertIn(record.id, context)
        self.assertIn("Pre-table text:", context)
        self.assertIn("Table:", context)
        self.assertIn("Post-table text:", context)
        self.assertIn("net cash from operating activities", context)

    def test_chat_messages_preserve_record_context_and_history(self) -> None:
        dataset = load_dataset()
        record = dataset.train[0]
        history = [ChatTurn(user="what is net cash in 2009?", assistant="206588")]

        messages = build_chat_messages(record, history, "what about 2008?")

        self.assertEqual(messages[0]["role"], "system")
        self.assertIn(record.id, messages[0]["content"])
        self.assertIn("using only the selected convfinqa record", build_system_prompt(record).lower())
        self.assertEqual(messages[1], {"role": "user", "content": "what is net cash in 2009?"})
        self.assertEqual(messages[2], {"role": "assistant", "content": "206588"})
        self.assertEqual(messages[3], {"role": "user", "content": "what about 2008?"})


if __name__ == "__main__":
    unittest.main()
