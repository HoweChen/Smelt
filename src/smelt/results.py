"""Case results: each then-expectation yields a score; the case aggregates pass/fail."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any


def _std(scores: list[float] | tuple[float, ...]) -> float:
    """Population std of the run scores; 0.0 for a single run."""
    return statistics.pstdev(scores) if len(scores) > 1 else 0.0


@dataclass(frozen=True)
class ExpectationResult:
    """Evaluation result of a single then-expectation.

    With repeated sampling (times > 1), ``score`` is the mean across runs and
    ``run_scores`` holds each individual run's score.
    """

    name: str  # expectation description, e.g. tool_call(run_command)
    score: float  # 0.0 ~ 1.0 (mean across runs when repeated)
    threshold: float  # pass threshold
    message: str = ""  # human-readable note (failure reason, etc.)
    details: dict[str, Any] = field(default_factory=dict)
    runs: int = 1  # how many runs produced this result
    run_scores: tuple[float, ...] = ()  # per-run scores; empty when runs == 1

    @property
    def passed(self) -> bool:
        return self.score >= self.threshold

    @property
    def score_std(self) -> float:
        """Spread across runs — large std means the score is unstable."""
        return _std(self.run_scores)

    def __str__(self) -> str:
        mark = "✔" if self.passed else "✘"
        base = f"{mark} {self.name}  score={self.score:.2f} (threshold={self.threshold:.2f})"
        if self.runs > 1:
            base += f"  ±{self.score_std:.2f} n={self.runs}"
        return f"{base}  — {self.message}" if self.message else base


@dataclass
class CaseResult:
    """Full result of one case.

    With repeated sampling (times > 1), ``score`` is the mean of per-run case
    scores (a run that errored counts as 0), ``run_scores`` lists each run, and
    ``run_errors`` records the runs that crashed.
    """

    case_name: str
    expectations: list[ExpectationResult]
    trace: Any = None  # smelt.trace.Trace; may be None on error
    error: str | None = None
    workspace: str | None = None
    report_path: str | None = None  # set when the case ran via .report()
    runs: int = 1  # how many times the case was executed
    run_scores: list[float] = field(default_factory=list)  # per-run case scores; empty when runs == 1
    run_errors: list[str] = field(default_factory=list)  # errors of crashed runs (partial failures)
    run_passed: list[bool] = field(default_factory=list)  # per-run pass flags; empty when runs == 1

    @property
    def score(self) -> float:
        """Case score: mean of run scores when repeated; otherwise mean of
        expectation scores; 1.0 when empty (unless errored)."""
        if self.error is not None:
            return 0.0
        if self.run_scores:
            return sum(self.run_scores) / len(self.run_scores)
        if not self.expectations:
            return 1.0
        return sum(e.score for e in self.expectations) / len(self.expectations)

    @property
    def score_std(self) -> float:
        """Spread of per-run case scores — large std means the result is unstable."""
        return _std(self.run_scores)

    @property
    def pass_hat(self) -> bool | None:
        """pass^k reliability (tau-bench): True iff EVERY run passed.

        The mean flatters an agent that succeeds sometimes; pass^k answers
        "can this skill be trusted every single time". None for a single run.
        """
        if not self.run_passed:
            return None
        return all(self.run_passed)

    @property
    def passed(self) -> bool:
        return self.error is None and all(e.passed for e in self.expectations)

    def summary(self) -> str:
        mark = "✔" if self.passed else "✘"
        base = f"{mark} case {self.case_name!r}  score={self.score:.2f}"
        if self.runs > 1:
            base += f" ±{self.score_std:.2f} n={self.runs}"
            base += f"  pass^{self.runs} {'✔' if self.pass_hat else '✘'}"
        lines = [base]
        if self.error is not None:
            lines.append(f"  error: {self.error}")
        for e in self.run_errors:
            lines.append(f"  run error: {e}")
        lines.extend(f"  {e}" for e in self.expectations)
        return "\n".join(lines)

    def assert_passed(self) -> CaseResult:
        """Pytest integration: raise AssertionError with the full summary on failure."""
        if not self.passed:
            raise AssertionError(self.summary())
        return self
