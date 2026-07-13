"""Run persistence, answer scoring, and result breakdowns.

The project separates model execution from scoring. A run JSONL stores only raw
model predictions. Evaluation later joins those saved predictions back to the
dataset, parses one answer string at a time, compares against strict
`executed_answers`, and builds Table 4 / Figure 5-style summaries.

This module owns parsing, normalization, and correctness checks. The model call
itself lives in `src.answers`.
"""

from __future__ import annotations

import json
import math
import random
import re
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Union

from pydantic import BaseModel, ConfigDict

from src.models import ConvFinQADataset, ConvFinQARecord
from src.prompts import ChatTurn

AnswerValue = Union[float, int, str]
AnswerFn = Callable[[ConvFinQARecord, list[ChatTurn], str], str]
"""Function signature used to keep evaluation independent from OpenAI calls."""

_NUMBER_PATTERN = re.compile(r"[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)")
_FINAL_ANSWER_PATTERN = re.compile(
    r"final\s+answer\s*:\s*(?P<answer>[^\n\r]+)",
    flags=re.IGNORECASE,
)
_PERCENT_QUESTION_PATTERN = re.compile(r"\b(?:percent|percentage|portion|ratio)\b", flags=re.IGNORECASE)


class NormalizedAnswer(BaseModel):
    """Comparable answer value used by evaluation.

    Percentage strings are normalized to their ratio value:
    `-3.3%` becomes `-0.033`.
    """

    model_config = ConfigDict(extra="forbid")

    raw: str
    value: float | str
    is_numeric: bool
    is_percent: bool = False


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


class TurnRunResult(BaseModel):
    """Raw LLM output for one conversation turn before scoring."""

    model_config = ConfigDict(extra="forbid")

    record_id: str
    turn_index: int
    question: str
    prediction: str


TurnResultCallback = Callable[[TurnRunResult], None]
"""Optional hook used by the CLI to checkpoint completed turns immediately."""


class BaselineRunSummary(BaseModel):
    """Aggregate result for a model run before scoring."""

    model_config = ConfigDict(extra="forbid")

    total_turns: int
    results: list[TurnRunResult]


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


def run_records(
    records: Sequence[ConvFinQARecord],
    answer_fn: AnswerFn,
    max_records: int | None = None,
    max_turns_per_record: int | None = None,
) -> BaselineRunSummary:
    """Replay gold conversation turns and save raw model answers."""
    results: list[TurnRunResult] = []

    for record in _limit_records(records, max_records):
        results.extend(run_record(record, answer_fn, max_turns_per_record))

    return summarize_run_results(results)


def run_records_parallel(
    records: Sequence[ConvFinQARecord],
    answer_fn: AnswerFn,
    max_records: int | None = None,
    max_turns_per_record: int | None = None,
    workers: int = 4,
) -> BaselineRunSummary:
    """Run records concurrently while keeping turns sequential per record."""
    selected_records = _limit_records(records, max_records)
    if workers <= 1:
        return run_records(
            records=selected_records,
            answer_fn=answer_fn,
            max_records=max_records,
            max_turns_per_record=max_turns_per_record,
        )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Parallelize only across records. Turns inside one record must stay
        # sequential because later questions may depend on prior model answers.
        # executor.map preserves selected_records order for stable JSONL output.
        record_results = list(
            executor.map(
                lambda record: run_record(record, answer_fn, max_turns_per_record),
                selected_records,
            ),
        )

    results = [result for record_result in record_results for result in record_result]
    return summarize_run_results(results)


def run_record(
    record: ConvFinQARecord,
    answer_fn: AnswerFn,
    max_turns_per_record: int | None = None,
    on_turn_result: TurnResultCallback | None = None,
) -> list[TurnRunResult]:
    """Replay one record's turns sequentially and return raw model answers."""
    # History is reset per record, matching the interactive chat behavior.
    history: list[ChatTurn] = []
    results: list[TurnRunResult] = []
    turn_count = _aligned_turn_count(record)
    if max_turns_per_record is not None:
        turn_count = min(turn_count, max_turns_per_record)

    for turn_index in range(turn_count):
        question = record.dialogue.conv_questions[turn_index]
        prediction = answer_fn(record, history, question)
        result = TurnRunResult(
            record_id=record.id,
            turn_index=turn_index,
            question=question,
            prediction=prediction,
        )
        results.append(result)
        if on_turn_result is not None:
            on_turn_result(result)
        history.append(ChatTurn(user=question, assistant=prediction))

    return results


def parse_answer_text(answer: str, question: str | None = None) -> NormalizedAnswer:
    """Parse an answer string into a comparable value when possible."""
    answer_text = _extract_final_answer(answer)
    normalized_text = _normalize_text(answer_text)
    matches = list(_NUMBER_PATTERN.finditer(normalized_text))
    match = matches[-1] if matches else None
    if match is None:
        return NormalizedAnswer(raw=answer_text, value=normalized_text, is_numeric=False)

    value = float(match.group().replace(",", ""))
    is_percent = "%" in normalized_text or _implied_percent_answer(
        full_answer=answer,
        final_answer=answer_text,
        question=question,
    )
    if is_percent:
        # Store percentages on the same scale as ConvFinQA executed answers.
        value = value / 100

    return NormalizedAnswer(raw=answer_text, value=value, is_numeric=True, is_percent=is_percent)


def normalize_gold_answer(executed_answer: AnswerValue, conv_answer: str) -> NormalizedAnswer:
    """Normalize gold fields with the conversational answer as display aid.

    `conv_answer` preserves display intent, such as percentages. If it cannot be
    parsed numerically, fall back to `executed_answer`. Use this for diagnostic
    display-normalized scoring, not for the paper-aligned headline metric.
    """
    parsed_conv_answer = parse_answer_text(conv_answer)
    if parsed_conv_answer.is_numeric:
        # Prefer display answer scale: "-3.3%" should compare as -0.033.
        return parsed_conv_answer

    if isinstance(executed_answer, (float, int)):
        # Some answers are not display-formatted, so the raw execution result is
        # the best comparable value.
        return NormalizedAnswer(
            raw=str(executed_answer),
            value=float(executed_answer),
            is_numeric=True,
        )

    return parse_answer_text(str(executed_answer))


def normalize_executed_answer(executed_answer: AnswerValue) -> NormalizedAnswer:
    """Normalize the executable gold result for strict execution accuracy."""
    if isinstance(executed_answer, (float, int)):
        return NormalizedAnswer(
            raw=str(executed_answer),
            value=float(executed_answer),
            is_numeric=True,
        )

    return parse_answer_text(str(executed_answer))


def answers_match(
    prediction: str,
    executed_answer: AnswerValue,
    question: str | None = None,
    rel_tol: float = 1e-3,
    abs_tol: float = 1e-3,
) -> bool:
    """Compare a predicted answer with executable gold only."""
    parsed_prediction = parse_answer_text(prediction, question=question)
    gold_answer = normalize_executed_answer(executed_answer)

    if _normalized_answers_match(
        parsed_prediction=parsed_prediction,
        gold_answer=gold_answer,
        rel_tol=rel_tol,
        abs_tol=abs_tol,
    ):
        return True

    if parsed_prediction.is_percent and question is not None and "%" not in _extract_final_answer(prediction):
        # Some dataset execution results keep percentage points as plain values
        # (for example 18 instead of 0.18). If inferred-percent parsing does
        # not match the executable gold, fall back to the literal final answer.
        return _normalized_answers_match(
            parsed_prediction=parse_answer_text(prediction),
            gold_answer=gold_answer,
            rel_tol=rel_tol,
            abs_tol=abs_tol,
        )

    return False


def display_answers_match(
    prediction: str,
    executed_answer: AnswerValue,
    conv_answer: str,
    question: str | None = None,
    rel_tol: float = 1e-3,
    abs_tol: float = 1e-3,
) -> bool:
    """Compare a prediction with display-normalized gold for diagnostics."""
    parsed_prediction = parse_answer_text(prediction, question=question)
    gold_answer = normalize_gold_answer(executed_answer, conv_answer)

    return _normalized_answers_match(
        parsed_prediction=parsed_prediction,
        gold_answer=gold_answer,
        rel_tol=rel_tol,
        abs_tol=abs_tol,
    )


def evaluate_run_results(
    run_results: Sequence[TurnRunResult],
    dataset: ConvFinQADataset,
) -> BaselineEvaluationSummary:
    """Score saved model predictions against strict executed answers."""
    records_by_id = {record.id: record for record in [*dataset.train, *dataset.dev]}
    evaluation_results: list[TurnEvaluationResult] = []

    for run_result in deduplicate_run_results(run_results):
        record = records_by_id[run_result.record_id]
        turn_index = run_result.turn_index
        gold_conv_answer = record.dialogue.conv_answers[turn_index]
        gold_executed_answer = record.dialogue.executed_answers[turn_index]
        parsed_prediction = parse_answer_text(run_result.prediction, question=run_result.question)
        normalized_gold = normalize_executed_answer(gold_executed_answer)

        evaluation_results.append(
            TurnEvaluationResult(
                record_id=run_result.record_id,
                turn_index=turn_index,
                question=run_result.question,
                prediction=run_result.prediction,
                prediction_value=parsed_prediction.value,
                prediction_is_percent=parsed_prediction.is_percent,
                gold_conv_answer=gold_conv_answer,
                gold_executed_answer=gold_executed_answer,
                gold_value=normalized_gold.value,
                gold_is_percent=normalized_gold.is_percent,
                is_correct=answers_match(
                    prediction=run_result.prediction,
                    executed_answer=gold_executed_answer,
                    question=run_result.question,
                ),
            ),
        )

    return summarize_results(evaluation_results)


def summarize_run_results(results: Sequence[TurnRunResult]) -> BaselineRunSummary:
    """Build an aggregate summary from raw run results."""
    return BaselineRunSummary(total_turns=len(results), results=list(results))


def summarize_results(results: Sequence[TurnEvaluationResult]) -> BaselineEvaluationSummary:
    """Build an aggregate summary from turn-level results."""
    correct_turns = sum(result.is_correct for result in results)
    total_turns = len(results)
    accuracy = correct_turns / total_turns if total_turns else 0.0

    return BaselineEvaluationSummary(
        total_turns=total_turns,
        correct_turns=correct_turns,
        accuracy=accuracy,
        results=list(results),
    )


def deduplicate_run_results(run_results: Sequence[TurnRunResult]) -> list[TurnRunResult]:
    """Keep the latest row for each record/turn while preserving first-seen order."""
    ordered_keys: list[tuple[str, int]] = []
    by_key: dict[tuple[str, int], TurnRunResult] = {}
    for result in run_results:
        key = (result.record_id, result.turn_index)
        if key not in by_key:
            ordered_keys.append(key)
        by_key[key] = result
    return [by_key[key] for key in ordered_keys]


def select_records(
    records: Sequence[ConvFinQARecord],
    max_records: int | None = None,
    random_seed: int | None = None,
) -> list[ConvFinQARecord]:
    """Select records in file order or by reproducible random sample."""
    selected_records = list(records)
    if random_seed is not None:
        rng = random.Random(random_seed)
        rng.shuffle(selected_records)
    return _limit_records(selected_records, max_records)


def _limit_records(records: Sequence[ConvFinQARecord], max_records: int | None) -> list[ConvFinQARecord]:
    """Return all records unless the caller explicitly provides a limit."""
    selected_records = list(records)
    if max_records is None:
        return selected_records
    return selected_records[:max_records]


def write_run_jsonl(summary: BaselineRunSummary, output_path: Path) -> None:
    """Write raw turn-level model outputs as JSONL for later scoring."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as file:
        for result in summary.results:
            file.write(json.dumps(result.model_dump()) + "\n")


def append_run_result_jsonl(result: TurnRunResult, output_path: Path) -> None:
    """Append one completed raw turn result for crash-resistant runs."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a") as file:
        file.write(json.dumps(result.model_dump()) + "\n")
        file.flush()


def write_selected_records_jsonl(records: Sequence[ConvFinQARecord], output_path: Path) -> None:
    """Write the selected record IDs/order used by a reproducible batch run."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as file:
        for index, record in enumerate(records):
            file.write(json.dumps({"index": index, "record_id": record.id}) + "\n")


def load_selected_record_ids_jsonl(records_path: Path) -> list[str]:
    """Load record IDs from a selected-record sidecar file."""
    return [
        json.loads(line)["record_id"]
        for line in records_path.read_text().splitlines()
        if line.strip()
    ]


def load_run_jsonl(results_path: Path) -> list[TurnRunResult]:
    """Load saved raw turn-level model outputs."""
    return [
        TurnRunResult.model_validate_json(line)
        for line in results_path.read_text().splitlines()
        if line.strip()
    ]


def write_results_jsonl(summary: BaselineEvaluationSummary, output_path: Path) -> None:
    """Write scored turn-level evaluation results as JSONL."""
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


def _normalized_answers_match(
    parsed_prediction: NormalizedAnswer,
    gold_answer: NormalizedAnswer,
    rel_tol: float,
    abs_tol: float,
) -> bool:
    """Compare two normalized answers with numeric tolerance when possible."""
    if parsed_prediction.is_numeric and gold_answer.is_numeric:
        assert isinstance(parsed_prediction.value, float)
        assert isinstance(gold_answer.value, float)
        return math.isclose(
            parsed_prediction.value,
            gold_answer.value,
            rel_tol=rel_tol,
            abs_tol=abs_tol,
        )

    return str(parsed_prediction.value).strip().lower() == str(gold_answer.value).strip().lower()


def _normalize_text(text: str) -> str:
    """Normalize common model wording before numeric parsing."""
    return (
        text.strip()
        .lower()
        .replace("negative ", "-")
        .replace("negative", "-")
        .replace("positive", "")
        .replace("−", "-")
    )


def _extract_final_answer(text: str) -> str:
    """Prefer an explicit final-answer line when the model provides one."""
    match = _FINAL_ANSWER_PATTERN.search(text)
    if match is None:
        return text
    return match.group("answer").strip()


def _implied_percent_answer(
    full_answer: str,
    final_answer: str,
    question: str | None,
) -> bool:
    """Infer a missing percent sign from the question and calculation text."""
    # Keep this intentionally narrow: it fixes false negatives where the model
    # calculated a percentage but omitted "%" on the machine-readable final line.
    # It does not use conv_answers and it should not turn ordinary numbers into
    # percentages unless the question and calculation both support that reading.
    if "%" in final_answer:
        return False
    if question is None or _PERCENT_QUESTION_PATTERN.search(question) is None:
        return False

    final_matches = list(_NUMBER_PATTERN.finditer(_normalize_text(final_answer)))
    if not final_matches:
        return False
    final_value = float(final_matches[-1].group().replace(",", ""))
    normalized_answer = _normalize_text(full_answer)
    if re.search(r"\bcalculation\s*:", normalized_answer) is None:
        return False

    percent_values = [
        float(match.group(1).replace(",", ""))
        for match in re.finditer(r"([-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+))\s*%", normalized_answer)
    ]
    if any(math.isclose(final_value, percent_value, rel_tol=1e-2, abs_tol=0.2) for percent_value in percent_values):
        return True

    return bool(re.search(r"(?:\*|x|×)\s*100\b", normalized_answer))
