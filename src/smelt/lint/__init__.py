"""smelt.lint — static quality linting for SKILL.md (the former skillcheck capability)."""

from smelt.lint.checks import ALL_CHECKS, run_checks
from smelt.lint.loader import discover_skills, load_skill
from smelt.lint.scorer import build_report, grade_of

__all__ = [
    "ALL_CHECKS",
    "build_report",
    "discover_skills",
    "grade_of",
    "load_skill",
    "run_checks",
]
