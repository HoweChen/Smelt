"""pytest plugin: provides the `smelt` fixture — run cases with full failure summaries.

Usage::

    def test_commit(smelt):
        smelt.check(
            new_case("commit")
            .given(fixed_agent("done"))
            .when(text("hi"))
            .then(output_equals("done"))
        )
"""

from __future__ import annotations

import pytest

from smelt.case import SmeltCase
from smelt.results import CaseResult


class SmeltSession:
    """Case executor for one test session; collects every case run."""

    def __init__(self) -> None:
        self.results: list[CaseResult] = []

    def run(self, case: SmeltCase) -> CaseResult:
        result = case.run()
        self.results.append(result)
        return result

    def check(self, case: SmeltCase) -> CaseResult:
        """Run and raise an AssertionError with the summary on failure; return CaseResult on success."""
        return self.run(case).assert_passed()


@pytest.fixture
def smelt() -> SmeltSession:
    return SmeltSession()
