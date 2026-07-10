"""Small baseline evaluation helpers.

The evaluator intentionally stays small: it replays gold questions, delegates
answer generation to a caller-provided function, and compares predictions with
the same normalization used by the chat prototype.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from pydantic import BaseModel, ConfigDict

from src.answers import answers_match
from src.models import ConvFinQARecord
from src.prompts import ChatTurn

AnswerFn = Callable[[ConvFinQARecord, list[ChatTurn], str], str]
"""Function signature used to keep evaluation independent from OpenAI calls."""


class TurnEvaluationResult(BaseModel):
    """Evaluation result for one conversation turn."""

    model_config = ConfigDict(extra="forbid")

    record_id: str
    turn_index: int
    question: str
    prediction: str
    gold_conv_answer: str
    gold_executed_answer: float | int | str
    is_correct: bool


class BaselineEvaluationSummary(BaseModel):
    """Aggregate result for a small baseline run."""

    model_config = ConfigDict(extra="forbid")

    total_turns: int
    correct_turns: int
    accuracy: float
    results: list[TurnEvaluationResult]


def evaluate_records(
    records: Sequence[ConvFinQARecord],
    answer_fn: AnswerFn,
    max_records: int,
    max_turns_per_record: int | None = None,
) -> BaselineEvaluationSummary:
    """Replay gold conversation turns and compare answers."""
    results: list[TurnEvaluationResult] = []

    for record in records[:max_records]:
        # History is reset per record, matching the interactive chat behavior.
        history: list[ChatTurn] = []
        turn_count = _aligned_turn_count(record)
        if max_turns_per_record is not None:
            turn_count = min(turn_count, max_turns_per_record)

        for turn_index in range(turn_count):
            question = record.dialogue.conv_questions[turn_index]
            prediction = answer_fn(record, history, question)
            gold_conv_answer = record.dialogue.conv_answers[turn_index]
            gold_executed_answer = record.dialogue.executed_answers[turn_index]
            # Compare against both gold fields because display format and raw
            # execution scale differ for many percentage answers.
            is_correct = answers_match(
                prediction=prediction,
                executed_answer=gold_executed_answer,
                conv_answer=gold_conv_answer,
            )

            results.append(
                TurnEvaluationResult(
                    record_id=record.id,
                    turn_index=turn_index,
                    question=question,
                    prediction=prediction,
                    gold_conv_answer=gold_conv_answer,
                    gold_executed_answer=gold_executed_answer,
                    is_correct=is_correct,
                ),
            )
            history.append(ChatTurn(user=question, assistant=prediction))

    correct_turns = sum(result.is_correct for result in results)
    total_turns = len(results)
    accuracy = correct_turns / total_turns if total_turns else 0.0

    return BaselineEvaluationSummary(
        total_turns=total_turns,
        correct_turns=correct_turns,
        accuracy=accuracy,
        results=results,
    )


def _aligned_turn_count(record: ConvFinQARecord) -> int:
    """Use only turns with every gold field present.

    A small number of cleaned records still have uneven dialogue field lengths.
    Taking the minimum keeps evaluation robust without mutating the dataset.
    """
    return min(
        len(record.dialogue.conv_questions),
        len(record.dialogue.conv_answers),
        len(record.dialogue.executed_answers),
    )
