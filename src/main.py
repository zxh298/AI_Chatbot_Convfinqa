"""Command-line entry points for the ConvFinQA prototype."""

import os
import sys
import threading
from pathlib import Path
from typing import Optional, cast

import typer
from dotenv import load_dotenv
from openai import APIError, OpenAI
from openai.types.chat import ChatCompletionMessageParam
from rich import print as rich_print

from src.data import find_record, load_dataset
from src.evaluation import (
    build_table4_breakdown,
    evaluate_records,
    evaluate_records_parallel,
    load_results_jsonl,
    select_records,
    write_results_jsonl,
)
from src.evidence import (
    EvidenceSnippet,
    build_evidence_snippets,
    build_rerank_messages,
    select_candidate_snippets,
    select_reranked_snippets,
)
from src.models import ConvFinQARecord
from src.prompts import ChatTurn, build_chat_messages

load_dotenv()

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
    use_evidence: bool = typer.Option(False, "--use-evidence", help="Use record-local evidence selection."),
    show_evidence: bool = typer.Option(False, "--show-evidence", help="Print selected evidence snippets before each answer."),
) -> None:
    """Ask questions about a specific ConvFinQA record."""
    # Load the selected record before touching the API, so invalid IDs fail fast
    # without requiring an OpenAI key or network call.
    dataset = load_dataset()
    located_record = find_record(dataset, record_id)
    if located_record is None:
        rich_print(f"[red]Error: record ID not found: {record_id}[/red]")
        raise typer.Exit(code=1)

    record = located_record.record
    rich_print(f"[green]Loaded {located_record.split} record:[/green] {record.id}")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        rich_print("Error: OPENAI_API_KEY not found")
        sys.exit(1)

    openai_client = OpenAI(api_key=api_key)
    history: list[ChatTurn] = []
    record_snippets = build_evidence_snippets(record) if use_evidence else []

    while True:
        message = input(">>> ")

        if message.strip().lower() in {"exit", "quit"}:
            break

        if not message.strip():
            rich_print("Empty message, please enter a valid question.")
            continue

        try:
            evidence_snippets = (
                _select_evidence_snippets(
                    openai_client=openai_client,
                    model="gpt-4o-mini",
                    record_snippets=record_snippets,
                    history=history,
                    question=message,
                )
                if use_evidence
                else []
            )
            if show_evidence and not use_evidence:
                rich_print("[yellow]--show-evidence requires --use-evidence to select snippets.[/yellow]")
            if show_evidence and evidence_snippets:
                rich_print("[magenta][bold]selected evidence:[/bold][/magenta]")
                for snippet in evidence_snippets:
                    rich_print(f"[magenta]- [{snippet.snippet_id}] {snippet.text}[/magenta]")

            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=cast(
                    list[ChatCompletionMessageParam],
                    build_chat_messages(record, history, message, evidence_snippets),
                ),
            ).choices[0].message.content

            rich_print(f"[blue][bold]assistant:[/bold] {response}[/blue]")

        except APIError as e:
            rich_print(f"[red]Error from OpenAI API: {e}")
            continue

        history.append(ChatTurn(user=message, assistant=response or ""))


@app.command()
def eval_baseline(
    split: str = typer.Option("dev", help="Dataset split to evaluate: train or dev."),
    max_records: int = typer.Option(3, help="Number of records to evaluate."),
    max_turns: Optional[int] = typer.Option(None, help="Optional maximum turns per record."),
    random_seed: Optional[int] = typer.Option(None, help="Random seed for reproducible record sampling."),
    model: str = typer.Option("gpt-4o", help="OpenAI model to use."),
    output_path: Optional[Path] = typer.Option(None, help="Optional JSONL path for turn-level results."),
    use_evidence: bool = typer.Option(False, "--use-evidence", help="Use record-local evidence selection."),
    workers: int = typer.Option(1, "--workers", min=1, help="Number of records to evaluate concurrently."),
) -> None:
    """Run strict executed-answer evaluation over train or dev records."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        rich_print("Error: OPENAI_API_KEY not found")
        raise typer.Exit(code=1)

    dataset = load_dataset()
    if split == "train":
        records = dataset.train
    elif split == "dev":
        records = dataset.dev
    else:
        rich_print("[red]Error: --split must be 'train' or 'dev'[/red]")
        raise typer.Exit(code=1)

    openai_client = OpenAI(api_key=api_key)
    snippet_cache: dict[str, list[EvidenceSnippet]] = {}
    snippet_cache_lock = threading.Lock()

    def answer_question(
        record: ConvFinQARecord,
        history: list[ChatTurn],
        question: str,
    ) -> str:
        # The evaluator owns replay/history; this nested function owns only the
        # model call. That keeps evaluation testable with fake answer functions.
        evidence_snippets: list[EvidenceSnippet] = []
        if use_evidence:
            # With --workers, multiple records can be evaluated at once. Guard
            # the cache so each record's deterministic snippets are built once.
            with snippet_cache_lock:
                record_snippets = snippet_cache.setdefault(record.id, build_evidence_snippets(record))
            evidence_snippets = _select_evidence_snippets(
                openai_client=openai_client,
                model=model,
                record_snippets=record_snippets,
                history=history,
                question=question,
            )
        response = openai_client.chat.completions.create(
            model=model,
            messages=cast(
                list[ChatCompletionMessageParam],
                build_chat_messages(record, history, question, evidence_snippets),
            ),
        ).choices[0].message.content
        return response or ""

    try:
        selected_records = select_records(records, max_records=max_records, random_seed=random_seed)
        # workers > 1 evaluates records concurrently, but each record still
        # replays its turns in order to preserve conversational dependencies.
        summary = evaluate_records(
            records=selected_records,
            answer_fn=answer_question,
            max_records=max_records,
            max_turns_per_record=max_turns,
        ) if workers == 1 else evaluate_records_parallel(
            records=selected_records,
            answer_fn=answer_question,
            max_records=max_records,
            max_turns_per_record=max_turns,
            workers=workers,
        )
    except APIError as e:
        rich_print(f"[red]Error from OpenAI API: {e}")
        raise typer.Exit(code=1) from e

    rich_print(
        f"[bold]Baseline {split} accuracy:[/bold] "
        f"{summary.correct_turns}/{summary.total_turns} "
        f"({summary.accuracy:.1%})",
    )
    if output_path is not None:
        write_results_jsonl(summary, output_path)
        rich_print(f"[green]Saved turn-level results to:[/green] {output_path}")

    for result in summary.results:
        status = "PASS" if result.is_correct else "FAIL"
        rich_print(
            f"{status} {result.record_id} turn {result.turn_index + 1}: "
            f"pred={result.prediction!r} gold={result.gold_conv_answer!r}",
        )


@app.command()
def analyze_results(
    results_path: Path = typer.Argument(..., help="JSONL file produced by eval-baseline."),
) -> None:
    """Print Table 4-style breakdowns for saved evaluation results."""
    dataset = load_dataset()
    results = load_results_jsonl(results_path)
    breakdown = build_table4_breakdown(results, dataset)

    rich_print(f"[bold]Breakdown for:[/bold] {results_path}")
    for row in breakdown:
        rich_print(
            f"{row.label}: "
            f"{row.correct_turns}/{row.total_turns} "
            f"({row.accuracy:.1%})",
        )


def _select_evidence_snippets(
    openai_client: OpenAI,
    model: str,
    record_snippets: list[EvidenceSnippet],
    history: list[ChatTurn],
    question: str,
) -> list[EvidenceSnippet]:
    """Select focused evidence with lexical filtering plus LLM reranking."""
    candidates = select_candidate_snippets(
        snippets=record_snippets,
        history=history,
        current_question=question,
        limit=30,
    )
    if not candidates:
        return []

    reranker_response = openai_client.chat.completions.create(
        model=model,
        messages=cast(
            list[ChatCompletionMessageParam],
            build_rerank_messages(
                history=history,
                current_question=question,
                candidate_snippets=candidates,
            ),
        ),
    ).choices[0].message.content
    return select_reranked_snippets(candidates, reranker_response or "")


if __name__ == "__main__":
    app()
