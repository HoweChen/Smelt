"""Base class for lint checks."""

from __future__ import annotations

from abc import ABC, abstractmethod

from smelt.lint.models import CheckResult, Message, SkillDoc

PASS_THRESHOLD = 60.0


class Check(ABC):
    """Base class of all checks. Subclasses only implement run()."""

    check_id: str = ""
    name: str = ""
    default_weight: float = 1.0

    @abstractmethod
    def run(self, skill: SkillDoc) -> CheckResult: ...

    def _result(self, score: float, messages: list[Message]) -> CheckResult:
        score = max(0.0, min(100.0, score))
        return CheckResult(
            check_id=self.check_id,
            name=self.name,
            score=score,
            passed=score >= PASS_THRESHOLD,
            messages=messages,
        )
