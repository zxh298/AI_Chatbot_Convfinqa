"""Tests for answer-version feature routing."""

# ruff: noqa: D102

from __future__ import annotations

import unittest

from src.answers import (
    _OFFLINE_FALLBACK_VERSIONS,
    _STRUCTURED_CALCULATION_VERSIONS,
    AnswerVersion,
)


class AnswerVersionRoutingTests(unittest.TestCase):
    def test_v5_and_v5a_use_structured_calculation(self) -> None:
        self.assertIn(AnswerVersion.V5, _STRUCTURED_CALCULATION_VERSIONS)
        self.assertIn(AnswerVersion.V5A, _STRUCTURED_CALCULATION_VERSIONS)

    def test_only_v5a_uses_offline_fallback(self) -> None:
        self.assertNotIn(AnswerVersion.V5, _OFFLINE_FALLBACK_VERSIONS)
        self.assertIn(AnswerVersion.V5A, _OFFLINE_FALLBACK_VERSIONS)


if __name__ == "__main__":
    unittest.main()
