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

from pydantic import BaseModel, ConfigDict, Field, ValidationError

PlanArgument: TypeAlias = Union[float, int, str]

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

    id: int = Field(ge=0)
    op: CalculationOp
    args: list[PlanArgument]


class CalculationPlan(BaseModel):
    """Validated calculation plan emitted by the model."""

    model_config = ConfigDict(extra="forbid")

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
    """Execute the first valid JSON calculation plan found in an answer."""
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
    values_by_id: dict[int, float] = {}
    for step in plan.steps:
        if step.id in values_by_id:
            raise ValueError(f"Duplicate calculation step id: {step.id}")
        values_by_id[step.id] = _execute_step(step, values_by_id)

    return _resolve_argument(plan.answer, values_by_id)


def _extract_calculation_plan(answer_text: str) -> CalculationPlan | None:
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


def _execute_step(step: CalculationStep, values_by_id: dict[int, float]) -> float:
    args = [_resolve_argument(arg, values_by_id) for arg in step.args]
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


def _resolve_argument(argument: PlanArgument, values_by_id: dict[int, float]) -> float:
    if isinstance(argument, (float, int)):
        return float(argument)
    if argument.startswith("#"):
        try:
            step_id = int(argument[1:])
        except ValueError as error:
            raise ValueError(f"Invalid step reference: {argument}") from error
        if step_id not in values_by_id:
            raise ValueError(f"Unknown step reference: {argument}")
        return values_by_id[step_id]
    try:
        return float(argument.replace(",", ""))
    except ValueError as error:
        raise ValueError(f"Non-numeric calculation argument: {argument}") from error


def _require_arg_count(step: CalculationStep, args: list[float], expected_count: int) -> None:
    if len(args) != expected_count:
        raise ValueError(f"{step.op.value} requires {expected_count} argument(s), got {len(args)}")


def _replace_final_answer(answer_text: str, value: str) -> str:
    replacement = f"Final answer: {value}"
    if _FINAL_ANSWER_PATTERN.search(answer_text):
        return _FINAL_ANSWER_PATTERN.sub(replacement, answer_text, count=1)
    return f"{answer_text.rstrip()}\n{replacement}"


def _format_number(value: float) -> str:
    if math.isclose(value, round(value), rel_tol=0.0, abs_tol=1e-10):
        return str(int(round(value)))
    return f"{value:.10g}"
