"""Answer generation for selected ConvFinQA records.

This module is the only place that calls the OpenAI chat API for final answers.
It turns a record, conversation history, and current question into raw model
prediction text. Evaluation-specific parsing and correctness checks live in
`src.evaluation`, not here.

Versions are explicit:
- v1 uses the full selected record as context.
- v2 adds record-local evidence selection before answering.
- v3 adds one no-gold verification retry on top of v2.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Sequence
from enum import Enum
from typing import cast

from openai import OpenAI, RateLimitError
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, ConfigDict, Field

from src.evidence import (
    EvidenceSnippet,
    build_evidence_snippets,
    select_relevant_evidence,
)
from src.logger import get_logger
from src.models import ConvFinQARecord
from src.prompts import ChatTurn, build_chat_messages
from src.verification import build_retry_instruction, verify_answer

logger = get_logger(__name__)
_MAX_RATE_LIMIT_RETRIES = 6
_INITIAL_RETRY_DELAY_SECONDS = 1.0
_MAX_RETRY_DELAY_SECONDS = 30.0


class AnswerVersion(str, Enum):
    """Available answer-generation versions."""

    V1 = "v1"
    V2 = "v2"
    V3 = "v3"


_EVIDENCE_SELECTION_VERSIONS = {AnswerVersion.V2, AnswerVersion.V3}
_VERIFICATION_VERSIONS = {AnswerVersion.V3}


class AnswerResponse(BaseModel):
    """One model answer plus the evidence snippets used to produce it."""

    model_config = ConfigDict(extra="forbid")

    text: str
    evidence_snippets: list[EvidenceSnippet] = Field(default_factory=list)


class OpenAIAnswerer:
    """Generate answers while hiding OpenAI and evidence-selection details."""

    def __init__(
        self,
        api_key: str,
        model: str,
        version: AnswerVersion = AnswerVersion.V1,
    ) -> None:
        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._version = version
        self._snippet_cache: dict[str, list[EvidenceSnippet]] = {}
        self._snippet_cache_lock = threading.Lock()

    def answer(
        self,
        record: ConvFinQARecord,
        history: Sequence[ChatTurn],
        question: str,
    ) -> AnswerResponse:
        """Answer one question for a selected record."""
        evidence_snippets = self._select_evidence(record, history, question)
        messages = build_chat_messages(
            record=record,
            history=list(history),
            current_question=question,
            evidence_snippets=evidence_snippets,
        )
        response_text = self._request_chat_completion(messages)
        response_text = self._retry_if_verification_fails(
            messages=messages,
            question=question,
            response_text=response_text,
            evidence_snippets=evidence_snippets,
        )
        return AnswerResponse(text=response_text, evidence_snippets=evidence_snippets)

    def _select_evidence(
        self,
        record: ConvFinQARecord,
        history: Sequence[ChatTurn],
        question: str,
    ) -> list[EvidenceSnippet]:
        if self._version not in _EVIDENCE_SELECTION_VERSIONS:
            return []

        # Multiple records may be processed concurrently in batch runs. Guard the
        # deterministic snippet cache so each record is built once per run.
        with self._snippet_cache_lock:
            record_snippets = self._snippet_cache.setdefault(record.id, build_evidence_snippets(record))

        return select_relevant_evidence(
            snippets=record_snippets,
            history=history,
            current_question=question,
            rerank_fn=self._request_chat_completion,
        )

    def _retry_if_verification_fails(
        self,
        messages: list[dict[str, str]],
        question: str,
        response_text: str,
        evidence_snippets: Sequence[EvidenceSnippet],
    ) -> str:
        if self._version not in _VERIFICATION_VERSIONS:
            return response_text

        verification = verify_answer(
            question=question,
            answer=response_text,
            evidence_snippets=evidence_snippets,
        )
        if not verification.should_retry or verification.reason is None:
            return response_text

        logger.info("Retrying answer after verification warning: %s", verification.reason)
        retry_messages = [
            *messages,
            {"role": "assistant", "content": response_text},
            {"role": "user", "content": build_retry_instruction(verification.reason)},
        ]
        return self._request_chat_completion(retry_messages)

    def _request_chat_completion(self, messages: list[dict[str, str]]) -> str:
        """Call the chat API and normalize an empty response to an empty string."""
        retry_delay = _INITIAL_RETRY_DELAY_SECONDS
        for attempt in range(_MAX_RATE_LIMIT_RETRIES + 1):
            try:
                return (
                    self._client.chat.completions.create(
                        model=self._model,
                        messages=cast(list[ChatCompletionMessageParam], messages),
                    ).choices[0].message.content
                    or ""
                )
            except RateLimitError:
                if attempt >= _MAX_RATE_LIMIT_RETRIES:
                    raise
                logger.warning(
                    "OpenAI rate limit reached; retrying in %.1fs (attempt %s/%s)",
                    retry_delay,
                    attempt + 1,
                    _MAX_RATE_LIMIT_RETRIES,
                )
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, _MAX_RETRY_DELAY_SECONDS)

        raise RuntimeError("unreachable retry state")
