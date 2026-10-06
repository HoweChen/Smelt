"""Clarity check: placeholders, unfinished markers, length control."""

from __future__ import annotations

import re

from smelt.lint.checks.base import Check
from smelt.lint.models import CheckResult, Message, Severity, SkillDoc

PLACEHOLDERS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\{\{.*?\}\}"), "template placeholder {{...}}"),
    (re.compile(r"\b(TODO|FIXME|TBD|XXX)\b"), "unfinished marker (TODO/FIXME/TBD/XXX)"),
    (re.compile(r"<[a-z ]*(your|placeholder|example)[^>]*>", re.IGNORECASE), "example placeholder tag"),
]

MAX_WORDS = 3000


class ClarityCheck(Check):
    check_id = "clarity"
    name = "Clarity (placeholders / length)"
    default_weight = 1.0

    def run(self, skill: SkillDoc) -> CheckResult:
        messages: list[Message] = []
        score = 100.0

        found: list[str] = []
        for pat, label in PLACEHOLDERS:
            if pat.search(skill.body):
                found.append(label)
        if found:
            messages.append(Message(Severity.ERROR, f"found unfilled content: {', '.join(found)}"))
            score -= 15 * len(found)

        if skill.word_count > MAX_WORDS:
            messages.append(
                Message(
                    Severity.WARNING,
                    f"body is ~{skill.word_count} words, exceeding {MAX_WORDS}; skills should stay concise — "
                    "move details into references/ or other companion files",
                )
            )
            score -= 15

        return self._result(score, messages)
