"""
Main typer app for ConvFinQA
"""

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI, APIError
import typer
from rich import print as rich_print

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
    
    # start openai client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        rich_print("Error: OPENAI_API_KEY not found")
        sys.exit(1)

    openai_client = OpenAI(api_key=api_key)
    history = []

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
                messages=[
                    {"role": "system", "content": f"You are a helpful assistant for record {record_id}."},
                    *[
                        {"role": "user", "content": h["user"]} if "assistant" not in h else
                        {"role": "assistant", "content": h["assistant"]}
                        for h in history
                    ],
                    {"role": "user", "content": message}
                ]
            ).choices[0].message.content

            rich_print(f"[blue][bold]assistant:[/bold] {response}[/blue]")
            
        
        except APIError as e:
            rich_print(f"[red]Error from OpenAI API: {e}")
            continue
        
        try:
            history.append({"user": message, "assistant": response})
        except Exception as e:
            rich_print(f"[red]Error appending to history: {e}")
            continue


@app.command()
def myfunc() -> None:
    """My hello world function"""
    # TODO: YOUR CODE HERE
    rich_print("Hello World")


if __name__ == "__main__":
    app()
