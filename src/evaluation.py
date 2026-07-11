"""Small baseline evaluation helpers.

The evaluator intentionally stays small: it replays gold questions, delegates
answer generation to a caller-provided function, and compares predictions with
the same normalization used by the chat prototype.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from src.answers import answers_match, normalize_executed_answer, parse_answer_text
from src.models import ConvFinQADataset, ConvFinQARecord
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
    prediction_value: float | str
    prediction_is_percent: bool
    gold_conv_answer: str
    gold_executed_answer: float | int | str
    gold_value: float | str
    gold_is_percent: bool
    is_correct: bool


class BaselineEvaluationSummary(BaseModel):
    """Aggregate result for a small baseline run."""

    model_config = ConfigDict(extra="forbid")

    total_turns: int
    correct_turns: int
    accuracy: float
    results: list[TurnEvaluationResult]


class BreakdownRow(BaseModel):
    """Accuracy row for one evaluation slice."""

    model_config = ConfigDict(extra="forbid")

    label: str
    correct_turns: int
    total_turns: int
    accuracy: float


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
            parsed_prediction = parse_answer_text(prediction)
            normalized_gold = normalize_executed_answer(gold_executed_answer)
            is_correct = answers_match(
                prediction=prediction,
                executed_answer=gold_executed_answer,
            )

            results.append(
                TurnEvaluationResult(
                    record_id=record.id,
                    turn_index=turn_index,
                    question=question,
                    prediction=prediction,
                    prediction_value=parsed_prediction.value,
                    prediction_is_percent=parsed_prediction.is_percent,
                    gold_conv_answer=gold_conv_answer,
                    gold_executed_answer=gold_executed_answer,
                    gold_value=normalized_gold.value,
                    gold_is_percent=normalized_gold.is_percent,
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


def select_records(
    records: Sequence[ConvFinQARecord],
    max_records: int,
    random_seed: int | None = None,
) -> list[ConvFinQARecord]:
    """Select records in file order or by reproducible random sample."""
    selected_records = list(records)
    if random_seed is not None:
        rng = random.Random(random_seed)
        rng.shuffle(selected_records)
    return selected_records[:max_records]


def write_results_jsonl(summary: BaselineEvaluationSummary, output_path: Path) -> None:
    """Write turn-level evaluation results as JSONL for later analysis."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as file:
        for result in summary.results:
            file.write(json.dumps(result.model_dump()) + "\n")


def load_results_jsonl(results_path: Path) -> list[TurnEvaluationResult]:
    """Load saved turn-level evaluation results."""
    return [
        TurnEvaluationResult.model_validate_json(line)
        for line in results_path.read_text().splitlines()
        if line.strip()
    ]


def build_table4_breakdown(
    results: Sequence[TurnEvaluationResult],
    dataset: ConvFinQADataset,
) -> list[BreakdownRow]:
    """Build Table 4-style breakdowns without rerunning the model."""
    records_by_id = {record.id: record for record in [*dataset.train, *dataset.dev]}
    groups: dict[str, list[TurnEvaluationResult]] = {
        "full results": list(results),
        "Number selection questions": [],
        "Program questions": [],
        "Simple conversations": [],
        "Hybrid conversations": [],
        "Hybrid conversations (first part)": [],
        "Hybrid conversations (second part)": [],
    }
    turn_groups: dict[int, list[TurnEvaluationResult]] = {}

    for result in results:
        record = records_by_id.get(result.record_id)
        if record is None:
            continue

        program = record.dialogue.turn_program[result.turn_index]
        if _is_number_selection_program(program):
            groups["Number selection questions"].append(result)
        else:
            groups["Program questions"].append(result)

        if record.features.has_type2_question:
            groups["Hybrid conversations"].append(result)
            if record.dialogue.qa_split[result.turn_index]:
                groups["Hybrid conversations (second part)"].append(result)
            else:
                groups["Hybrid conversations (first part)"].append(result)
        else:
            groups["Simple conversations"].append(result)

        turn_groups.setdefault(result.turn_index, []).append(result)

    rows = [_make_breakdown_row(label, label_results) for label, label_results in groups.items()]
    rows.extend(
        _make_breakdown_row(f"Turn {turn_index}", turn_results)
        for turn_index, turn_results in sorted(turn_groups.items())
    )
    return rows


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


def _is_number_selection_program(program: str) -> bool:
    """A turn is number selection when its gold program is already a value."""
    return "(" not in program and ")" not in program


def _make_breakdown_row(label: str, results: Sequence[TurnEvaluationResult]) -> BreakdownRow:
    """Summarize one group of turn-level results."""
    correct_turns = sum(result.is_correct for result in results)
    total_turns = len(results)
    accuracy = correct_turns / total_turns if total_turns else 0.0
    return BreakdownRow(
        label=label,
        correct_turns=correct_turns,
        total_turns=total_turns,
        accuracy=accuracy,
    )
