"""CLI entry points for the ConvFinQA prototype.

This file stays intentionally thin: it parses command options, loads dataset
records, delegates answer generation to `src.answers`, delegates scoring to
`src.evaluation`, and prints user-facing output. Core model, evidence, parsing,
and evaluation logic should live in their own modules.
"""

import os
import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from openai import APIError
from rich import print as rich_print

from src import evaluation
from src.answers import AnswerVersion, OpenAIAnswerer
from src.data import find_record, load_dataset
from src.example_retrieval import record_document_key
from src.logger import get_logger
from src.models import ConvFinQADataset, ConvFinQARecord
from src.prompts import ChatTurn

load_dotenv()
logger = get_logger(__name__)

# Keep the CLI as the only user-facing surface for now. The implementation lives
# in small modules so each piece can be tested without invoking Typer/OpenAI.
app = typer.Typer(
    name="main",
    help="ConvFinQA record-aware chat and evaluation CLI.",
    add_completion=True,
    no_args_is_help=True,
)


@app.command()
def chat(
    record_id: str = typer.Argument(..., help="ID of the record to chat about"),
    version: AnswerVersion = typer.Option(
        AnswerVersion.V1,
        "--version",
        help="Answering version: v1=full-record baseline, v2=evidence selection, v3=evidence plus verification retry, v4=v3 plus train-example retrieval.",
    ),
    show_evidence: bool = typer.Option(False, "--show-evidence", help="Print selected evidence snippets before each answer."),
    show_examples: bool = typer.Option(False, "--show-examples", help="Print v4 retrieved train examples before each answer."),
) -> None:
    """Ask questions about a specific ConvFinQA record."""
    # Load the selected record before touching the API, so invalid IDs fail fast
    # without requiring an OpenAI key or network call.
    dataset = load_dataset()
    located_record = find_record(dataset, record_id)
    if located_record is None:
        rich_print(f"[red]Error: record ID not found: {record_id}[/red]")
        logger.warning("Record ID not found: %s", record_id)
        raise typer.Exit(code=1)

    record = located_record.record
    rich_print(f"[green]Loaded {located_record.split} record:[/green] {record.id}")
    logger.info("Loaded %s record: %s", located_record.split, record.id)

    answerer = OpenAIAnswerer(
        api_key=_require_api_key(),
        model="gpt-4o-mini",
        version=version,
        example_records=dataset.train,
    )
    history: list[ChatTurn] = []

    while True:
        message = input(">>> ")

        if message.strip().lower() in {"exit", "quit"}:
            break

        if not message.strip():
            rich_print("Empty message, please enter a valid question.")
            continue

        try:
            answer = answerer.answer(record=record, history=history, question=message)
            if show_evidence and version not in {AnswerVersion.V2, AnswerVersion.V3, AnswerVersion.V4}:
                rich_print("[yellow]--show-evidence requires --version v2, v3, or v4 to select snippets.[/yellow]")
            if show_evidence and answer.evidence_snippets:
                rich_print("[magenta][bold]selected evidence:[/bold][/magenta]")
                for snippet in answer.evidence_snippets:
                    rich_print(f"[magenta]- [{snippet.snippet_id}] {snippet.text}[/magenta]")
            if show_examples and version is not AnswerVersion.V4:
                rich_print("[yellow]--show-examples requires --version v4.[/yellow]")
            if show_examples and answer.reasoning_examples:
                rich_print("[cyan][bold]similar train examples:[/bold][/cyan]")
                for example in answer.reasoning_examples:
                    rich_print(
                        "[cyan]"
                        f"- {example.record_id} turn {example.turn_index}: "
                        f"Q: {example.question} | "
                        f"program: {example.turn_program} | "
                        f"answer: {example.conv_answer}"
                        "[/cyan]",
                    )

            rich_print(f"[blue][bold]assistant:[/bold] {answer.text}[/blue]")

        except APIError as e:
            rich_print(f"[red]Error from OpenAI API: {e}")
            logger.exception("OpenAI API error during chat")
            continue

        history.append(ChatTurn(user=message, assistant=evaluation.answer_for_history(answer.text)))


@app.command("run")
def run(
    split: str = typer.Option("dev", help="Dataset split to evaluate: train or dev."),
    max_records: Optional[int] = typer.Option(None, help="Optional maximum number of records to run."),
    max_turns: Optional[int] = typer.Option(None, help="Optional maximum turns per record."),
    random_seed: Optional[int] = typer.Option(None, help="Random seed for reproducible record sampling."),
    model: str = typer.Option("gpt-4o", help="OpenAI model to use."),
    output_path: Path = typer.Option(..., help="JSONL path for raw turn-level model outputs."),
    selected_records_path: Optional[Path] = typer.Option(
        None,
        help="Optional JSONL path for the selected record IDs/order. Defaults to <output>_records.jsonl.",
    ),
    resume: bool = typer.Option(False, "--resume", help="Resume a checkpointed run by rerunning only incomplete records."),
    version: AnswerVersion = typer.Option(
        AnswerVersion.V1,
        "--version",
        help="Answering version: v1=full-record baseline, v2=evidence selection, v3=evidence plus verification retry, v4=v3 plus train-example retrieval.",
    ),
    workers: int = typer.Option(1, "--workers", min=1, help="Number of records to run concurrently."),
) -> None:
    """Run the model over train or dev records and save raw predictions."""
    dataset = load_dataset()
    records = _records_for_split(dataset, split)

    logger.info(
        "Starting run: split=%s model=%s version=%s max_records=%s max_turns=%s seed=%s workers=%s output=%s",
        split,
        model,
        version.value,
        max_records,
        max_turns,
        random_seed,
        workers,
        output_path,
    )
    try:
        selected_records_path = selected_records_path or _selected_records_path_for(output_path)
        selected_records = _selected_records_for_run(
            records=records,
            selected_records_path=selected_records_path,
            max_records=max_records,
            random_seed=random_seed,
            resume=resume,
        )
        if not resume:
            evaluation.write_selected_records_jsonl(selected_records, selected_records_path)
            _reset_output_file(output_path)
        elif not output_path.exists():
            raise typer.BadParameter(f"--resume requires existing output file: {output_path}")

        records_to_run = (
            _incomplete_records_for_resume(
                selected_records=selected_records,
                output_path=output_path,
                max_turns_per_record=max_turns,
            )
            if resume
            else selected_records
        )

        if resume:
            rich_print(
                f"[cyan]resume:[/cyan] {len(records_to_run)}/{len(selected_records)} selected records are incomplete and will be rerun",
            )

        answerer = OpenAIAnswerer(
            api_key=_require_api_key(),
            model=model,
            version=version,
            example_records=_example_records_for_run(
                dataset=dataset,
                split=split,
                selected_records=selected_records,
                version=version,
            ),
        )

        def answer_question(
            record: ConvFinQARecord,
            history: list[ChatTurn],
            question: str,
        ) -> str:
            return answerer.answer(record, history, question).text

        # Workers > 1 runs records concurrently, but each record still replays
        # turns in order to preserve conversational dependencies. Turn rows are
        # appended immediately so a mid-run API failure does not lose everything.
        if workers == 1:
            summary = _run_records_with_checkpoints(
                records=records_to_run,
                answer_fn=answer_question,
                output_path=output_path,
                max_turns_per_record=max_turns,
            )
        else:
            summary = _run_records_parallel_with_checkpoints(
                records=records_to_run,
                answer_fn=answer_question,
                output_path=output_path,
                max_turns_per_record=max_turns,
                workers=workers,
            )
    except APIError as e:
        rich_print(f"[red]Error from OpenAI API: {e}")
        logger.exception("OpenAI API error during batch run")
        raise typer.Exit(code=1) from e

    logger.info("Saved raw predictions: %s (%s turns)", output_path, summary.total_turns)
    rich_print(f"[bold]{split} run complete:[/bold] {summary.total_turns} turns")
    rich_print(f"[green]Saved selected record IDs to:[/green] {selected_records_path}")
    rich_print(f"[green]Saved raw turn-level predictions to:[/green] {output_path}")

    _print_run_results(summary.results)


@app.command("evaluate")
def evaluate_results(
    run_path: Path = typer.Argument(..., help="Raw JSONL file produced by run."),
    output_path: Optional[Path] = typer.Option(None, help="Optional JSONL path for scored turn-level results."),
) -> None:
    """Score saved raw predictions against strict executed answers."""
    logger.info("Evaluating saved run: %s", run_path)
    dataset = load_dataset()
    run_results = evaluation.load_run_jsonl(run_path)
    summary = evaluation.evaluate_run_results(run_results, dataset)
    breakdown = evaluation.build_table4_breakdown(summary.results, dataset)

    rich_print(f"[bold]Evaluation for:[/bold] {run_path}")
    _print_breakdown(breakdown)

    if output_path is not None:
        evaluation.write_results_jsonl(summary, output_path)
        logger.info("Saved scored results: %s", output_path)
        rich_print(f"[green]Saved scored turn-level results to:[/green] {output_path}")


@app.command()
def analyze_results(
    results_path: Path = typer.Argument(..., help="Scored JSONL file produced by evaluate."),
) -> None:
    """Print Table 4-style breakdowns for saved scored evaluation results."""
    logger.info("Analyzing scored results: %s", results_path)
    dataset = load_dataset()
    results = evaluation.load_results_jsonl(results_path)
    breakdown = evaluation.build_table4_breakdown(results, dataset)

    rich_print(f"[bold]Breakdown for:[/bold] {results_path}")
    _print_breakdown(breakdown)


def _require_api_key() -> str:
    """Read the OpenAI API key or exit with a clear CLI error."""
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return api_key
    rich_print("Error: OPENAI_API_KEY not found")
    logger.error("OPENAI_API_KEY not found")
    raise typer.Exit(code=1)


def _records_for_split(
    dataset: ConvFinQADataset,
    split: str,
) -> list[ConvFinQARecord]:
    """Return the selected dataset split or exit on invalid input."""
    if split == "train":
        return dataset.train
    if split == "dev":
        return dataset.dev
    rich_print("[red]Error: --split must be 'train' or 'dev'[/red]")
    logger.warning("Invalid split requested: %s", split)
    raise typer.Exit(code=1)


def _selected_records_for_run(
    records: Sequence[ConvFinQARecord],
    selected_records_path: Path,
    max_records: int | None,
    random_seed: int | None,
    resume: bool,
) -> list[ConvFinQARecord]:
    """Select records for a fresh run or load the original selection on resume."""
    records_by_id = {record.id: record for record in records}
    if not resume:
        return evaluation.select_records(records, max_records=max_records, random_seed=random_seed)

    if not selected_records_path.exists():
        raise typer.BadParameter(f"--resume requires existing selected records file: {selected_records_path}")

    selected_ids = evaluation.load_selected_record_ids_jsonl(selected_records_path)
    missing_ids = [record_id for record_id in selected_ids if record_id not in records_by_id]
    if missing_ids:
        raise typer.BadParameter(f"Selected records file contains IDs outside --split: {missing_ids[:3]}")
    return [records_by_id[record_id] for record_id in selected_ids]


def _example_records_for_run(
    *,
    dataset: ConvFinQADataset,
    split: str,
    selected_records: Sequence[ConvFinQARecord],
    version: AnswerVersion,
) -> list[ConvFinQARecord]:
    """Return v4 retrieval examples without leaking evaluation records.

    For dev evaluation, train examples are allowed. For train-sample evaluation,
    remove the whole selected sample from the retrieval pool, including matching
    Single/Double variants that share the same underlying PDF page.
    """
    if version is not AnswerVersion.V4:
        return []
    if split != "train":
        return dataset.train

    selected_document_keys = {record_document_key(record.id) for record in selected_records}
    return [
        record
        for record in dataset.train
        if record_document_key(record.id) not in selected_document_keys
    ]


def _incomplete_records_for_resume(
    selected_records: Sequence[ConvFinQARecord],
    output_path: Path,
    max_turns_per_record: int | None,
) -> list[ConvFinQARecord]:
    """Return selected records that do not yet have all expected turn rows."""
    completed_turns_by_record: dict[str, set[int]] = {}
    for result in evaluation.load_run_jsonl(output_path):
        completed_turns_by_record.setdefault(result.record_id, set()).add(result.turn_index)

    incomplete_records: list[ConvFinQARecord] = []
    for record in selected_records:
        expected_turns = _expected_turn_count(record, max_turns_per_record)
        completed_turns = completed_turns_by_record.get(record.id, set())
        if any(turn_index not in completed_turns for turn_index in range(expected_turns)):
            incomplete_records.append(record)
    return incomplete_records


def _expected_turn_count(record: ConvFinQARecord, max_turns_per_record: int | None) -> int:
    """Expected evaluated turns for one record under the current run options."""
    turn_count = min(
        len(record.dialogue.conv_questions),
        len(record.dialogue.conv_answers),
        len(record.dialogue.executed_answers),
    )
    if max_turns_per_record is None:
        return turn_count
    return min(turn_count, max_turns_per_record)


def _run_records_with_checkpoints(
    records: Sequence[ConvFinQARecord],
    answer_fn: evaluation.AnswerFn,
    output_path: Path,
    max_turns_per_record: int | None,
) -> evaluation.BaselineRunSummary:
    """Run records sequentially while appending each completed turn."""
    results: list[evaluation.TurnRunResult] = []
    total_records = len(records)

    for completed_records, record in enumerate(records, start=1):
        record_results = evaluation.run_record(
            record=record,
            answer_fn=answer_fn,
            max_turns_per_record=max_turns_per_record,
            on_turn_result=lambda result: evaluation.append_run_result_jsonl(result, output_path),
        )
        results.extend(record_results)
        _print_record_progress(completed_records, total_records)

    return evaluation.summarize_run_results(results)


def _run_records_parallel_with_checkpoints(
    records: Sequence[ConvFinQARecord],
    answer_fn: evaluation.AnswerFn,
    output_path: Path,
    max_turns_per_record: int | None,
    workers: int,
) -> evaluation.BaselineRunSummary:
    """Run records concurrently, append completed turns, and print progress."""
    results_by_record: dict[str, list[evaluation.TurnRunResult]] = {}
    write_lock = threading.Lock()
    total_records = len(records)

    def save_turn(result: evaluation.TurnRunResult) -> None:
        with write_lock:
            evaluation.append_run_result_jsonl(result, output_path)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                evaluation.run_record,
                record,
                answer_fn,
                max_turns_per_record,
                save_turn,
            ): record
            for record in records
        }
        for completed_records, future in enumerate(as_completed(futures), start=1):
            record = futures[future]
            results_by_record[record.id] = future.result()
            _print_record_progress(completed_records, total_records)

    ordered_results = [
        result
        for record in records
        for result in results_by_record.get(record.id, [])
    ]
    return evaluation.summarize_run_results(ordered_results)


def _reset_output_file(output_path: Path) -> None:
    """Create an empty checkpoint file before starting a fresh run."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("")


def _selected_records_path_for(output_path: Path) -> Path:
    """Default sidecar path for the selected record IDs/order."""
    return output_path.with_name(f"{output_path.stem}_records.jsonl")


def _print_record_progress(completed_records: int, total_records: int) -> None:
    """Print record-level progress for long API runs."""
    rich_print(f"[cyan]progress:[/cyan] {completed_records}/{total_records} records completed")


def _print_breakdown(rows: Sequence[evaluation.BreakdownRow]) -> None:
    """Print Table 4/Figure 5-style rows."""
    for row in rows:
        rich_print(
            f"{row.label}: "
            f"{row.correct_turns}/{row.total_turns} "
            f"({row.accuracy:.1%})",
        )


def _print_run_results(results: Sequence[evaluation.TurnRunResult]) -> None:
    """Print raw predictions from a run."""
    for result in results:
        rich_print(
            f"{result.record_id} turn {result.turn_index + 1}: "
            f"pred={result.prediction!r}",
        )


if __name__ == "__main__":
    app()
