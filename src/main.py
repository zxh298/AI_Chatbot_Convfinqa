"""
Main typer app for ConvFinQA
"""

import os
import sys
from typing import cast

import typer
from dotenv import load_dotenv
from openai import APIError, OpenAI
from openai.types.chat import ChatCompletionMessageParam
from rich import print as rich_print

from src.data import find_record, load_dataset
from src.prompts import ChatTurn, build_chat_messages

load_dotenv()

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
    dataset = load_dataset()
    located_record = find_record(dataset, record_id)
    if located_record is None:
        rich_print(f"[red]Error: record ID not found: {record_id}[/red]")
        raise typer.Exit(code=1)

    record = located_record.record
    rich_print(f"[green]Loaded {located_record.split} record:[/green] {record.id}")

    # start openai client
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

        # response = "RESPONSE"
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
def myfunc() -> None:
    """My hello world function"""
    # TODO: YOUR CODE HERE
    rich_print("Hello World")


if __name__ == "__main__":
    app()
