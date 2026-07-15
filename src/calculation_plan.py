"""Structured calculation plans for deterministic v5 answer execution.

The model may still choose the relevant values and operation, but v5 asks it to
express that reasoning as a tiny JSON plan. This module validates the plan with
Pydantic and executes only a closed set of numeric operations, avoiding regex
parsing of free-form `Operation:` text for the final arithmetic.
"""

from __future__ import annotations

import json
import math
import re
from enum import Enum
from typing import TypeAlias, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

PlanArgument: TypeAlias = Union[float, int, str]
PlanId: TypeAlias = Union[int, str]

_FINAL_ANSWER_PATTERN = re.compile(
    r"final\s+answer\s*:\s*(?P<answer>[^\n\r]+)",
    flags=re.IGNORECASE,
)


class CalculationOp(str, Enum):
    """Small allowed operation set for ConvFinQA-style arithmetic."""

    SELECT = "select"
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"
    NEGATE = "negate"
    ABS = "abs"
    MAX = "max"
    MIN = "min"


class CalculationStep(BaseModel):
    """One executable step in a calculation plan."""

    model_config = ConfigDict(extra="forbid")

    id: PlanId
    op: CalculationOp
    args: list[PlanArgument]

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: PlanId) -> PlanId:
        """Keep step references readable and unambiguous."""
        if isinstance(value, int) and value < 0:
            raise ValueError("step id must be non-negative")
        if isinstance(value, str) and not value.strip():
            raise ValueError("step id must not be empty")
        return value


class CalculationValue(BaseModel):
    """One named source value extracted from evidence."""

    model_config = ConfigDict(extra="forbid")

    id: str
    value: float
    evidence: str | None = None

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        """Keep value references readable and unambiguous."""
        if not value.strip():
            raise ValueError("value id must not be empty")
        if value.startswith("#"):
            raise ValueError("value id must not start with #")
        return value


class CalculationPlan(BaseModel):
    """Validated calculation plan emitted by the model."""

    model_config = ConfigDict(extra="forbid")

    values: list[CalculationValue] = Field(default_factory=list)
    steps: list[CalculationStep]
    answer: PlanArgument


class PlanExecutionResult(BaseModel):
    """Result of trying to execute a model-proposed calculation plan."""

    model_config = ConfigDict(extra="forbid")

    executed: bool
    text: str
    value: float | None = None
    reason: str | None = None


def apply_calculation_plan(answer_text: str) -> PlanExecutionResult:
    """Execute the first valid JSON calculation plan found in an answer.

    This is v5's key step: the LLM still proposes values and operations, but
    code owns the final arithmetic and rewrites the `Final answer:` line with
    the executed result.
    """
    plan = _extract_calculation_plan(answer_text)
    if plan is None:
        return PlanExecutionResult(
            executed=False,
            text=answer_text,
            reason="No valid calculation plan JSON found.",
        )

    try:
        value = execute_calculation_plan(plan)
    except (ValueError, ZeroDivisionError) as error:
        return PlanExecutionResult(
            executed=False,
            text=answer_text,
            reason=str(error),
        )

    return PlanExecutionResult(
        executed=True,
        text=_replace_final_answer(answer_text, _format_number(value)),
        value=value,
    )


def execute_calculation_plan(plan: CalculationPlan) -> float:
    """Execute a validated calculation plan and return its numeric answer."""
    # Values and step outputs share one reference namespace. This catches plans
    # where a model accidentally reuses an id for two different meanings.
    values_by_reference: dict[str, float] = {}
    for value in plan.values:
        if value.id in values_by_reference:
            raise ValueError(f"Duplicate calculation value id: {value.id}")
        values_by_reference[value.id] = value.value

    for step in plan.steps:
        step_reference_keys = _reference_keys_for_step_id(step.id)
        if any(key in values_by_reference for key in step_reference_keys):
            raise ValueError(f"Duplicate calculation step id: {step.id}")
        step_value = _execute_step(step, values_by_reference)
        for key in step_reference_keys:
            values_by_reference[key] = step_value

    return _resolve_argument(plan.answer, values_by_reference)


def _extract_calculation_plan(answer_text: str) -> CalculationPlan | None:
    """Scan free-form model text for the first JSON object matching the schema."""
    decoder = json.JSONDecoder()
    for start_index, character in enumerate(answer_text):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(answer_text[start_index:])
        except json.JSONDecodeError:
            continue
        try:
            return CalculationPlan.model_validate(payload)
        except ValidationError:
            continue
    return None


def _reference_keys_for_step_id(step_id: PlanId) -> list[str]:
    if isinstance(step_id, int):
        return [str(step_id), f"#{step_id}"]
    return [step_id]


def _execute_step(step: CalculationStep, values_by_reference: dict[str, float]) -> float:
    """Execute one closed-set arithmetic operation."""
    args = [_resolve_argument(arg, values_by_reference) for arg in step.args]
    if step.op is CalculationOp.SELECT:
        _require_arg_count(step, args, 1)
        return args[0]
    if step.op is CalculationOp.ADD:
        _require_arg_count(step, args, 2)
        return args[0] + args[1]
    if step.op is CalculationOp.SUBTRACT:
        _require_arg_count(step, args, 2)
        return args[0] - args[1]
    if step.op is CalculationOp.MULTIPLY:
        _require_arg_count(step, args, 2)
        return args[0] * args[1]
    if step.op is CalculationOp.DIVIDE:
        _require_arg_count(step, args, 2)
        return args[0] / args[1]
    if step.op is CalculationOp.NEGATE:
        _require_arg_count(step, args, 1)
        return -args[0]
    if step.op is CalculationOp.ABS:
        _require_arg_count(step, args, 1)
        return abs(args[0])
    if step.op is CalculationOp.MAX:
        if not args:
            raise ValueError("max requires at least one argument")
        return max(args)
    if step.op is CalculationOp.MIN:
        if not args:
            raise ValueError("min requires at least one argument")
        return min(args)
    raise ValueError(f"Unsupported calculation op: {step.op}")


def _resolve_argument(argument: PlanArgument, values_by_reference: dict[str, float]) -> float:
    """Resolve literals, named values, and step references to numbers."""
    if isinstance(argument, (float, int)):
        return float(argument)
    if argument in values_by_reference:
        return values_by_reference[argument]
    if argument.startswith("#"):
        raise ValueError(f"Unknown step reference: {argument}")
    try:
        return float(argument.replace(",", ""))
    except ValueError as error:
        raise ValueError(f"Non-numeric calculation argument: {argument}") from error


def _require_arg_count(step: CalculationStep, args: list[float], expected_count: int) -> None:
    if len(args) != expected_count:
        raise ValueError(f"{step.op.value} requires {expected_count} argument(s), got {len(args)}")


def _replace_final_answer(answer_text: str, value: str) -> str:
    """Replace or append the answer line while preserving the model's audit trail."""
    replacement = f"Final answer: {value}"
    if _FINAL_ANSWER_PATTERN.search(answer_text):
        return _FINAL_ANSWER_PATTERN.sub(replacement, answer_text, count=1)
    return f"{answer_text.rstrip()}\n{replacement}"


def _format_number(value: float) -> str:
    if math.isclose(value, round(value), rel_tol=0.0, abs_tol=1e-10):
        return str(int(round(value)))
    return f"{value:.10g}"
