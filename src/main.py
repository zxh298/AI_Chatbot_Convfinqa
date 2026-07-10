"""Command-line entry points for the ConvFinQA prototype."""

import os
import sys
from typing import Optional, cast

import typer
from dotenv import load_dotenv
from openai import APIError, OpenAI
from openai.types.chat import ChatCompletionMessageParam
from rich import print as rich_print

from src.data import find_record, load_dataset
from src.evaluation import evaluate_records
from src.models import ConvFinQARecord
from src.prompts import ChatTurn, build_chat_messages

load_dotenv()

# Keep the CLI as the only user-facing surface for now. The implementation lives
# in small modules so each piece can be tested without invoking Typer/OpenAI.
app = typer.Typer(
    name="main",
    help="Boilerplate app for ConvFinQA",
    add_completion=True,
    no_args_is_help=True,
)


@app.command()
def chat(
    record_id: str = typer.Argument(..., help="ID of the record to chat about"),
) -> None:
    """Ask questions about a specific record"""
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

    while True:
        message = input(">>> ")

        if message.strip().lower() in {"exit", "quit"}:
            break

        if not message.strip():
            rich_print("Empty message, please enter a valid question.")
            continue

        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=cast(
                    list[ChatCompletionMessageParam],
                    build_chat_messages(record, history, message),
                ),
            ).choices[0].message.content

            rich_print(f"[blue][bold]assistant:[/bold] {response}[/blue]")

        except APIError as e:
            rich_print(f"[red]Error from OpenAI API: {e}")
            continue

        history.append(ChatTurn(user=message, assistant=response or ""))


@app.command()
def eval_baseline(
    max_records: int = typer.Option(3, help="Number of dev records to evaluate."),
    max_turns: Optional[int] = typer.Option(3, help="Maximum turns per record."),
    model: str = typer.Option("gpt-4o", help="OpenAI model to use."),
) -> None:
    """Run a small dev-set baseline evaluation."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        rich_print("Error: OPENAI_API_KEY not found")
        raise typer.Exit(code=1)

    dataset = load_dataset()
    openai_client = OpenAI(api_key=api_key)

    def answer_question(
        record: ConvFinQARecord,
        history: list[ChatTurn],
        question: str,
    ) -> str:
        # The evaluator owns replay/history; this nested function owns only the
        # model call. That keeps evaluation testable with fake answer functions.
        response = openai_client.chat.completions.create(
            model=model,
            messages=cast(
                list[ChatCompletionMessageParam],
                build_chat_messages(record, history, question),
            ),
        ).choices[0].message.content
        return response or ""

    try:
        summary = evaluate_records(
            records=dataset.dev,
            answer_fn=answer_question,
            max_records=max_records,
            max_turns_per_record=max_turns,
        )
    except APIError as e:
        rich_print(f"[red]Error from OpenAI API: {e}")
        raise typer.Exit(code=1) from e

    rich_print(
        f"[bold]Baseline dev accuracy:[/bold] "
        f"{summary.correct_turns}/{summary.total_turns} "
        f"({summary.accuracy:.1%})",
    )
    for result in summary.results:
        status = "PASS" if result.is_correct else "FAIL"
        rich_print(
            f"{status} {result.record_id} turn {result.turn_index + 1}: "
            f"pred={result.prediction!r} gold={result.gold_conv_answer!r}",
        )


@app.command()
def myfunc() -> None:
    """My hello world function"""
    # TODO: YOUR CODE HERE
    rich_print("Hello World")


if __name__ == "__main__":
    app()
