"""Test coverage check: does the skill ship with tests / examples / evals."""

from __future__ import annotations

from smelt.lint.checks.base import Check
from smelt.lint.models import CheckResult, Message, Severity, SkillDoc

EVIDENCE_DIRS = ("tests", "test", "examples", "evals", "eval")
EVIDENCE_FILES = ("eval.json", "eval.yaml", "test_cases.json")


class TestCoverageCheck(Check):
    check_id = "testcoverage"
    name = "Test coverage (cases / examples)"
    default_weight = 0.8

    def run(self, skill: SkillDoc) -> CheckResult:
        has_dir = any((skill.dir / d).is_dir() for d in EVIDENCE_DIRS)
        has_file = any((skill.dir / f).is_file() for f in EVIDENCE_FILES)
        # Inline examples in the body count as weak evidence
        has_inline_example = "## 示例" in skill.body or "## Example" in skill.body

        if has_dir or has_file:
            return self._result(100.0, [])
        if has_inline_example:
            return self._result(
                70.0,
                [
                    Message(
                        Severity.WARNING,
                        "only inline examples; add a standalone tests/ or evals/ directory with executable cases",
                        fix="add a tests/ or evals/ directory with executable cases",
                    )
                ],
            )
        return self._result(
            30.0,
            [
                Message(
                    Severity.ERROR,
                    "no test cases / examples / eval files; the skill's actual behavior cannot be verified",
                    fix="add at least one behavior test or eval case",
                )
            ],
        )
