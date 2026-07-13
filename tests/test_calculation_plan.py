"""Tests for v5 structured calculation plan execution."""

# ruff: noqa: D102

from __future__ import annotations

import unittest

from src.calculation_plan import (
    CalculationPlan,
    CalculationStep,
    apply_calculation_plan,
    execute_calculation_plan,
)


class CalculationPlanTests(unittest.TestCase):
    def test_executes_plan_with_step_references(self) -> None:
        plan = CalculationPlan(
            steps=[
                CalculationStep(id=0, op="subtract", args=[206588, 181001]),
                CalculationStep(id=1, op="divide", args=["#0", 181001]),
            ],
            answer="#1",
        )

        self.assertAlmostEqual(execute_calculation_plan(plan), 0.14136, places=5)

    def test_apply_plan_replaces_final_answer_with_executed_value(self) -> None:
        answer = (
            "Target: change over prior year\n"
            'Calculation plan JSON: {"steps":[{"id":0,"op":"subtract","args":[206588,181001]},'
            '{"id":1,"op":"divide","args":["#0",181001]}],"answer":"#1"}\n'
            "Final answer: 14.1%"
        )

        result = apply_calculation_plan(answer)

        self.assertTrue(result.executed)
        self.assertIn("Final answer: 0.14136", result.text)

    def test_invalid_plan_falls_back_to_original_answer(self) -> None:
        answer = 'Calculation plan JSON: {"steps":[{"id":0,"op":"divide","args":[1,0]}],"answer":"#0"}\nFinal answer: 0'

        result = apply_calculation_plan(answer)

        self.assertFalse(result.executed)
        self.assertEqual(result.text, answer)


if __name__ == "__main__":
    unittest.main()
