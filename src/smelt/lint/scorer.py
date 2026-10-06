"""Scoring and grading."""

from __future__ import annotations

from smelt.lint.checks import ALL_CHECKS
from smelt.lint.models import CheckResult, SkillDoc, SkillReport


def grade_of(score: float) -> str:
    """Score → grade. Below 60 is F (fail)."""
    for threshold, grade in ((90, "A"), (80, "B"), (70, "C"), (60, "D")):
        if score >= threshold:
            return grade
    return "F"


def build_report(
    skill: SkillDoc,
    results: list[CheckResult],
    weights: dict[str, float] | None = None,
) -> SkillReport:
    """Aggregate check scores by weight into a report.

    weights is keyed by check_id; checks without an entry use their default_weight.
    """
    if weights is None:
        weights = {c.check_id: c.default_weight for c in ALL_CHECKS}
    total_w = sum(weights.get(r.check_id, 1.0) for r in results)
    if total_w <= 0:
        total_w = 1.0
    total = sum(r.score * weights.get(r.check_id, 1.0) for r in results) / total_w
    total = round(total, 1)
    return SkillReport(
        skill_path=skill.dir,
        results=results,
        total_score=total,
        grade=grade_of(total),
    )
