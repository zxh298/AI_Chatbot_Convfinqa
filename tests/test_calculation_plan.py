"""Tests for v5 structured calculation plan execution."""

# ruff: noqa: D102

from __future__ import annotations

import unittest

from src.calculation_plan import (
    CalculationPlan,
    CalculationStep,
    CalculationValue,
    apply_calculation_plan,
    execute_calculation_plan,
)


class CalculationPlanTests(unittest.TestCase):
    def test_executes_auditable_plan_with_named_values(self) -> None:
        plan = CalculationPlan(
            values=[
                CalculationValue(id="drawn_amount", value=4.7, evidence="E3"),
                CalculationValue(id="facility_amount", value=150, evidence="E1"),
            ],
            steps=[
                CalculationStep(id="ratio", op="divide", args=["drawn_amount", "facility_amount"]),
            ],
            answer="ratio",
        )

        self.assertAlmostEqual(execute_calculation_plan(plan), 0.0313333333, places=8)

    def test_executes_direct_selection_plan_with_named_value(self) -> None:
        plan = CalculationPlan(
            values=[CalculationValue(id="credit_facility_amount", value=150, evidence="T-31")],
            steps=[CalculationStep(id="answer", op="select", args=["credit_facility_amount"])],
            answer="answer",
        )

        self.assertEqual(execute_calculation_plan(plan), 150)

    def test_executes_plan_with_step_references(self) -> None:
        plan = CalculationPlan(
            steps=[
                CalculationStep(id=0, op="subtract", args=[206588, 181001]),
                CalculationStep(id=1, op="divide", args=["#0", 181001]),
            ],
            answer="#1",
        )

        self.assertAlmostEqual(execute_calculation_plan(plan), 0.14136, places=5)

    def test_apply_auditable_plan_replaces_final_answer_with_executed_value(self) -> None:
        answer = (
            "Target: percentage of credit facility represented by drawn amount\n"
            "Values:\n"
            "- drawn_amount = 4.7 from evidence E3\n"
            "- facility_amount = 150 from evidence E1\n"
            "Operation: drawn_amount / facility_amount\n"
            'Calculation plan JSON: {"values":[{"id":"drawn_amount","value":4.7,"evidence":"E3"},'
            '{"id":"facility_amount","value":150,"evidence":"E1"}],'
            '"steps":[{"id":"ratio","op":"divide","args":["drawn_amount","facility_amount"]}],'
            '"answer":"ratio"}\n'
            "Final answer: 3.1%"
        )

        result = apply_calculation_plan(answer)

        self.assertTrue(result.executed)
        self.assertIn("Final answer: 0.03133333333", result.text)

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

    def test_duplicate_named_value_and_step_reference_is_invalid(self) -> None:
        plan = CalculationPlan(
            values=[CalculationValue(id="ratio", value=1.0)],
            steps=[CalculationStep(id="ratio", op="divide", args=[4.7, 150])],
            answer="ratio",
        )

        with self.assertRaisesRegex(ValueError, "Duplicate calculation step id"):
            execute_calculation_plan(plan)


if __name__ == "__main__":
    unittest.main()
