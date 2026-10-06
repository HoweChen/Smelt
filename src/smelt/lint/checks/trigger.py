"""Trigger coverage check: can an agent tell when to use this skill."""

from __future__ import annotations

import re

from smelt.lint.checks.base import Check
from smelt.lint.models import CheckResult, Message, Severity, SkillDoc

TRIGGER_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"use (this|when)", re.IGNORECASE), "English trigger guidance (use this / use when)"),
    (re.compile(r"trigger", re.IGNORECASE), "trigger keyword"),
    (re.compile(r"当.{0,20}(时|请求|需要|要求)"), "Chinese trigger description (当…时)"),
    (re.compile(r"触发"), "'trigger' keyword (Chinese)"),
    (re.compile(r"本 skill|this skill", re.IGNORECASE), "skill self-reference"),
]


class TriggerCheck(Check):
    check_id = "trigger"
    name = "Trigger coverage (when to use)"
    default_weight = 1.5

    def run(self, skill: SkillDoc) -> CheckResult:
        haystack = skill.description + "\n" + skill.body
        hits = [label for pat, label in TRIGGER_PATTERNS if pat.search(haystack)]

        messages: list[Message] = []
        score = 100.0
        if not hits:
            messages.append(
                Message(Severity.ERROR, "no trigger guidance found (use when / 当…时 / trigger); agents won't know when to load this skill")
            )
            score = 0.0
        elif len(hits) <= 2:
            messages.append(
                Message(Severity.WARNING, f"weak trigger guidance (only {len(hits)} kind(s) matched: {', '.join(hits)})")
            )
            score -= 25

        return self._result(score, messages)
