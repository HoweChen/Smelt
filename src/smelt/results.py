"""Case results: each then-expectation yields a score; the case aggregates pass/fail."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ExpectationResult:
    """Evaluation result of a single then-expectation."""

    name: str  # expectation description, e.g. tool_call(run_command)
    score: float  # 0.0 ~ 1.0
    threshold: float  # pass threshold
    message: str = ""  # human-readable note (failure reason, etc.)
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.score >= self.threshold

    def __str__(self) -> str:
        mark = "✔" if self.passed else "✘"
        base = f"{mark} {self.name}  score={self.score:.2f} (threshold={self.threshold:.2f})"
        return f"{base}  — {self.message}" if self.message else base


@dataclass
class CaseResult:
    """Full result of one case."""

    case_name: str
    expectations: list[ExpectationResult]
    trace: Any = None  # smelt.trace.Trace; may be None on error
    error: str | None = None
    workspace: str | None = None

    @property
    def score(self) -> float:
        """Case score: mean of expectation scores; 1.0 when empty (unless errored)."""
        if self.error is not None:
            return 0.0
        if not self.expectations:
            return 1.0
        return sum(e.score for e in self.expectations) / len(self.expectations)

    @property
    def passed(self) -> bool:
        return self.error is None and all(e.passed for e in self.expectations)

    def summary(self) -> str:
        mark = "✔" if self.passed else "✘"
        lines = [f"{mark} case {self.case_name!r}  score={self.score:.2f}"]
        if self.error is not None:
            lines.append(f"  error: {self.error}")
        lines.extend(f"  {e}" for e in self.expectations)
        return "\n".join(lines)

    def assert_passed(self) -> CaseResult:
        """Pytest integration: raise AssertionError with the full summary on failure."""
        if not self.passed:
            raise AssertionError(self.summary())
        return self
