"""Tests for lightweight train-example retrieval."""

# ruff: noqa: D102

from __future__ import annotations

import unittest

from src.answers import AnswerResponse
from src.example_retrieval import (
    build_example_index,
    format_reasoning_examples,
    retrieve_reasoning_examples,
)
from src.models import ConvFinQARecord, Dialogue, Document, Features
from src.prompts import ChatTurn


class ExampleRetrievalTests(unittest.TestCase):
    def test_retrieves_similar_train_examples_and_excludes_current_record(self) -> None:
        current_record = _make_record(
            record_id="Double_ETR/2016/page_424.pdf",
            questions=["what percentage, then, of this total did that amount represent?"],
            programs=["divide(4.7, 150)"],
            answers=["3.1%"],
            executed=[0.03133],
        )
        similar_record = _make_record(
            record_id="Double_AMT/2010/page_111.pdf",
            questions=["what percentage, then, of this total did that amount represent?"],
            programs=["divide(774209, 1197607)"],
            answers=["64.6%"],
            executed=[0.64646],
        )
        unrelated_record = _make_record(
            record_id="Single_ETR/2016/page_424.pdf-1",
            questions=["what was the ending balance?"],
            programs=["150"],
            answers=["150"],
            executed=[150],
        )
        index = build_example_index([current_record, similar_record, unrelated_record])

        examples = retrieve_reasoning_examples(
            index=index,
            history=[],
            current_question="what percentage, then, of this total did that amount represent?",
            current_record_id=current_record.id,
            limit=2,
        )

        self.assertEqual(examples[0].record_id, "Double_AMT/2010/page_111.pdf")
        self.assertNotIn(current_record.id, {example.record_id for example in examples})
        self.assertNotIn(unrelated_record.id, {example.record_id for example in examples})

    def test_format_reasoning_examples_warns_not_to_copy_numbers(self) -> None:
        record = _make_record(
            record_id="similar",
            questions=["what is that value times 1000?"],
            programs=["multiply(12.5, const_1000)"],
            answers=["12500"],
            executed=[12500],
        )
        example = build_example_index([record])[0].example

        formatted = format_reasoning_examples([example])

        self.assertIn("Similar solved train examples:", formatted)
        self.assertIn("Do not copy their numbers", formatted)
        self.assertIn("multiply(12.5, const_1000)", formatted)

    def test_follow_up_history_contributes_to_retrieval(self) -> None:
        percentage_record = _make_record(
            record_id="percentage",
            questions=[
                "what was the facility amount?",
                "what percentage, then, did that amount represent?",
            ],
            programs=["150", "divide(4.7, 150)"],
            answers=["150", "3.1%"],
            executed=[150, 0.03133],
        )
        index = build_example_index([percentage_record])

        examples = retrieve_reasoning_examples(
            index=index,
            history=[ChatTurn(user="what was the drawn amount?", assistant="Final answer: 4.7")],
            current_question="what percentage, then, did that amount represent?",
            current_record_id="different-current-record",
            limit=1,
        )

        self.assertEqual(examples[0].record_id, "percentage")
        self.assertEqual(examples[0].turn_index, 1)

    def test_answer_response_can_carry_reasoning_examples_for_inspection(self) -> None:
        record = _make_record(
            record_id="similar",
            questions=["what percentage did this represent?"],
            programs=["divide(10, 100)"],
            answers=["10%"],
            executed=[0.1],
        )
        example = build_example_index([record])[0].example

        response = AnswerResponse(text="Final answer: 10%", reasoning_examples=[example])

        self.assertEqual(response.reasoning_examples[0].record_id, "similar")


def _make_record(
    *,
    record_id: str,
    questions: list[str],
    programs: list[str],
    answers: list[str],
    executed: list[float | int | str],
) -> ConvFinQARecord:
    return ConvFinQARecord(
        id=record_id,
        doc=Document(pre_text="text", table={"year": {"metric": 1}}, post_text="more text"),
        dialogue=Dialogue(
            conv_questions=questions,
            conv_answers=answers,
            turn_program=programs,
            executed_answers=executed,
            qa_split=[False for _question in questions],
        ),
        features=Features(
            num_dialogue_turns=len(questions),
            has_type2_question=False,
            has_duplicate_columns=False,
            has_non_numeric_values=False,
        ),
    )


if __name__ == "__main__":
    unittest.main()
