"""Prompt construction for the record-aware chat."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from src.formatting import format_record_context
from src.models import ConvFinQARecord


class ChatTurn(BaseModel):
    """One completed chat turn."""

    model_config = ConfigDict(extra="forbid")

    user: str
    assistant: str


def build_system_prompt(record: ConvFinQARecord) -> str:
    """Build the system prompt for a selected ConvFinQA record."""
    return "\n\n".join(
        [
            "You are a financial question answering assistant.",
            "Answer the user's questions using only the selected ConvFinQA record below.",
            "Use the conversation history when it is relevant to the current question.",
            "For numerical answers, include a brief calculation when arithmetic is needed.",
            "If the answer is not supported by the record, say that the record does not provide enough information.",
            format_record_context(record),
        ],
    )


def build_chat_messages(
    record: ConvFinQARecord,
    history: list[ChatTurn],
    current_question: str,
) -> list[dict[str, str]]:
    """Build OpenAI chat messages with record context and prior turns."""
    messages = [{"role": "system", "content": build_system_prompt(record)}]

    for turn in history:
        messages.append({"role": "user", "content": turn.user})
        messages.append({"role": "assistant", "content": turn.assistant})

    messages.append({"role": "user", "content": current_question})
    return messages
