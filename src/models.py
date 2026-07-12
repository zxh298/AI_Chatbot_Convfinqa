"""Pydantic models for the cleaned ConvFinQA dataset.

These models define the typed boundary between the raw JSON file and the rest
of the application. They intentionally mirror the dataset structure closely so
loading is explicit without adding unnecessary abstraction.
"""

from __future__ import annotations

from typing import Union

from pydantic import BaseModel, ConfigDict, Field

TableValue = Union[float, int, str]


class Document(BaseModel):
    """Financial document context for one ConvFinQA record."""

    model_config = ConfigDict(extra="forbid")

    pre_text: str
    post_text: str
    table: dict[str, dict[str, TableValue]]


class Dialogue(BaseModel):
    """Gold conversation data included with each record."""

    model_config = ConfigDict(extra="forbid")

    conv_questions: list[str]
    conv_answers: list[str]
    turn_program: list[str]
    executed_answers: list[TableValue]
    qa_split: list[bool]


class Features(BaseModel):
    """Tomoro-provided metadata about a cleaned ConvFinQA record."""

    model_config = ConfigDict(extra="forbid")

    num_dialogue_turns: int
    has_type2_question: bool
    has_duplicate_columns: bool
    has_non_numeric_values: bool


class ConvFinQARecord(BaseModel):
    """One cleaned ConvFinQA conversation over a financial document."""

    model_config = ConfigDict(extra="forbid")

    id: str
    doc: Document
    dialogue: Dialogue
    features: Features


class ConvFinQADataset(BaseModel):
    """Train/dev dataset container."""

    model_config = ConfigDict(extra="forbid")

    train: list[ConvFinQARecord] = Field(default_factory=list)
    dev: list[ConvFinQARecord] = Field(default_factory=list)
