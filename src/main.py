"""CLI entry points for the ConvFinQA prototype.

This file stays intentionally thin: it parses command options, loads dataset
records, delegates answer generation to `src.answers`, delegates scoring to
`src.evaluation`, and prints user-facing output. Core model, evidence, parsing,
and evaluation logic should live in their own modules.
"""

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from openai import APIError
from rich import print as rich_print

from src import evaluation
from src.answers import AnswerVersion, OpenAIAnswerer
from src.data import find_record, load_dataset
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
        help="Answering version: v1=full-record baseline, v2=evidence selection.",
    ),
    show_evidence: bool = typer.Option(False, "--show-evidence", help="Print selected evidence snippets before each answer."),
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

    answerer = OpenAIAnswerer(api_key=_require_api_key(), model="gpt-4o-mini", version=version)
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
            if show_evidence and version != AnswerVersion.V2:
                rich_print("[yellow]--show-evidence requires --version v2 to select snippets.[/yellow]")
            if show_evidence and answer.evidence_snippets:
                rich_print("[magenta][bold]selected evidence:[/bold][/magenta]")
                for snippet in answer.evidence_snippets:
                    rich_print(f"[magenta]- [{snippet.snippet_id}] {snippet.text}[/magenta]")

            rich_print(f"[blue][bold]assistant:[/bold] {answer.text}[/blue]")

        except APIError as e:
            rich_print(f"[red]Error from OpenAI API: {e}")
            logger.exception("OpenAI API error during chat")
            continue

        history.append(ChatTurn(user=message, assistant=answer.text))


@app.command("run")
def run_baseline(
    split: str = typer.Option("dev", help="Dataset split to evaluate: train or dev."),
    max_records: Optional[int] = typer.Option(None, help="Optional maximum number of records to run."),
    max_turns: Optional[int] = typer.Option(None, help="Optional maximum turns per record."),
    random_seed: Optional[int] = typer.Option(None, help="Random seed for reproducible record sampling."),
    model: str = typer.Option("gpt-4o", help="OpenAI model to use."),
    output_path: Path = typer.Option(..., help="JSONL path for raw turn-level model outputs."),
    version: AnswerVersion = typer.Option(
        AnswerVersion.V1,
        "--version",
        help="Answering version: v1=full-record baseline, v2=evidence selection.",
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
    answerer = OpenAIAnswerer(api_key=_require_api_key(), model=model, version=version)

    def answer_question(
        record: ConvFinQARecord,
        history: list[ChatTurn],
        question: str,
    ) -> str:
        return answerer.answer(record, history, question).text

    try:
        selected_records = evaluation.select_records(records, max_records=max_records, random_seed=random_seed)
        # workers > 1 runs records concurrently, but each record still
        # replays its turns in order to preserve conversational dependencies.
        summary = evaluation.run_records(
            records=selected_records,
            answer_fn=answer_question,
            max_records=max_records,
            max_turns_per_record=max_turns,
        ) if workers == 1 else evaluation.run_records_parallel(
            records=selected_records,
            answer_fn=answer_question,
            max_records=max_records,
            max_turns_per_record=max_turns,
            workers=workers,
        )
    except APIError as e:
        rich_print(f"[red]Error from OpenAI API: {e}")
        logger.exception("OpenAI API error during batch run")
        raise typer.Exit(code=1) from e

    evaluation.write_run_jsonl(summary, output_path)
    logger.info("Saved raw predictions: %s (%s turns)", output_path, summary.total_turns)
    rich_print(f"[bold]{split} run complete:[/bold] {summary.total_turns} turns")
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
