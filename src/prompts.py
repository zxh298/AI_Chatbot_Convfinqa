"""Prompt construction for record-aware answering.

The prompt stays intentionally direct: provide the selected record, preserve
chat history, include optional evidence snippets, and ask for a concise final
answer plus a lightweight calculation check. This module only builds messages;
it does not call the model or parse outputs.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from src.evidence import EvidenceSnippet, format_evidence_snippets
from src.example_retrieval import ReasoningExample, format_reasoning_examples
from src.formatting import format_record_context
from src.models import ConvFinQARecord


class ChatTurn(BaseModel):
    """One completed chat turn."""

    model_config = ConfigDict(extra="forbid")

    user: str
    assistant: str


def build_system_prompt(
    record: ConvFinQARecord,
    evidence_snippets: list[EvidenceSnippet] | None = None,
    reasoning_examples: list[ReasoningExample] | None = None,
    use_structured_calculation: bool = False,
) -> str:
    """Build the system prompt for a selected ConvFinQA record."""
    sections = [
        "You are a financial question answering assistant.",
        "Answer the user's questions using only the selected ConvFinQA record below.",
        "Use the conversation history when it is relevant to the current question.",
        "Use the relevant evidence section as a focus aid, but rely on the full record if the selected snippets are incomplete.",
        "Before giving the final answer, write a lightweight value check with these lines: `Target:`, `Values:`, and `Operation:`.",
        "`Target:` should name exactly what the current question asks for, especially when one evidence sentence contains multiple values.",
        "`Values:` should list the needed record values and any competing values from the same evidence sentence, with short labels such as `letters of credit outstanding = 4.7` and `cash borrowings = 0`.",
        "When the evidence contains multiple plausible values, select the value whose label and role best match the question target, not simply the first value or the zero value.",
        "`Operation:` should be `select <label>` for number-selection questions or a simple arithmetic expression for calculation questions.",
        "This value check is not a ConvFinQA DSL program; it is only a short guard against choosing the wrong number, denominator, sign, or direction.",
        "Then write `Final answer: <value>` where `<value>` is only the final comparable answer, without units, dates, or explanation.",
        "Use a separate `Calculation:` line after the final answer when arithmetic or units need to be explained.",
        "If the answer is not supported by the record, say that the record does not provide enough information.",
    ]
    if use_structured_calculation:
        sections.extend(
            [
                "For this version, also output a machine-readable `Calculation plan JSON:` block before the final answer.",
                'The JSON must have exactly this shape: {"steps":[{"id":0,"op":"subtract","args":[206588,181001]}],"answer":"#0"}.',
                "Allowed ops are `select`, `add`, `subtract`, `multiply`, `divide`, `negate`, `abs`, `max`, and `min`; reference prior steps as `#0`, `#1`, etc.",
                "For percentage, portion, ratio, or rate questions, prefer a ratio-scale plan such as `divide(part, total)` and let the final answer be the raw executable number, not a percentage display.",
                "The code will execute this JSON locally, so make the plan contain only the values and operations needed for the current question.",
                "For numerical answers, the final-answer value should be a pure executable number such as `3` or `-0.03264`, without units, words, dates, or `%`.",
            ],
        )
    else:
        sections.append(
            "For numerical answers, the final-answer value should match the dataset style: use 3 instead of $3 million, and use -3.3% instead of the raw decimal ratio -0.03264 when the question asks for a percentage.",
        )
    if reasoning_examples is not None:
        sections.extend(
            [
                "Use similar solved train examples only to understand the reasoning pattern. Do not copy their numbers; the current answer must come from the selected record.",
                format_reasoning_examples(reasoning_examples),
            ],
        )
    sections.extend(
        [
            format_evidence_snippets(evidence_snippets or []),
            format_record_context(record),
        ],
    )
    return "\n\n".join(sections)


def build_chat_messages(
    record: ConvFinQARecord,
    history: list[ChatTurn],
    current_question: str,
    evidence_snippets: list[EvidenceSnippet] | None = None,
    reasoning_examples: list[ReasoningExample] | None = None,
    use_structured_calculation: bool = False,
) -> list[dict[str, str]]:
    """Build OpenAI chat messages with record context and prior turns."""
    messages = [
        {
            "role": "system",
            "content": build_system_prompt(
                record,
                evidence_snippets,
                reasoning_examples,
                use_structured_calculation,
            ),
        },
    ]

    # Preserve previous turns as actual chat messages instead of flattening them
    # into the prompt, which better matches how users experience follow-ups.
    for turn in history:
        messages.append({"role": "user", "content": turn.user})
        messages.append({"role": "assistant", "content": turn.assistant})

    messages.append({"role": "user", "content": current_question})
    return messages
