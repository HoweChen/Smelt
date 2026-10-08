"""Clarity check: placeholders, unfinished markers, length control."""

from __future__ import annotations

import re

from smelt.lint.checks.base import Check
from smelt.lint.markdown import fences, prose_segments
from smelt.lint.models import CheckResult, Message, Severity, SkillDoc

# (pattern, label, skip_code): skip_code=True patterns are matched against
# prose only, so template syntax shown in code examples (e.g. Vue's
# {{ interpolation }}) is not mistaken for an unfilled placeholder
PLACEHOLDERS: list[tuple[re.Pattern, str, bool]] = [
    (re.compile(r"\{\{.*?\}\}"), "template placeholder {{...}}", True),
    (re.compile(r"\b(TODO|FIXME|TBD|XXX)\b"), "unfinished marker (TODO/FIXME/TBD/XXX)", False),
    (re.compile(r"<[a-z ]*(your|placeholder|example)[^>]*>", re.IGNORECASE), "example placeholder tag", False),
]

MAX_WORDS = 3000
MAX_LINES = 500
# Heuristic: a fence line starting with one of these tokens is treated as
# code. Prose can still start with these words (residual false positives are
# accepted and capped at 3 messages); plain-text fences like directory trees
# do not match.
CODE_LIKE_RE = re.compile(
    r"^\s*(def |class |import \S|from \S|function |const |let |var |#!|<template>|<script>|public |private )"
)


class ClarityCheck(Check):
    check_id = "clarity"
    name = "Clarity (placeholders / length)"
    default_weight = 1.0

    def run(self, skill: SkillDoc) -> CheckResult:
        messages: list[Message] = []
        score = 100.0

        found: list[str] = []
        prose = "\n".join(prose_segments(skill.body))
        for pat, label, skip_code in PLACEHOLDERS:
            haystack = prose if skip_code else skill.body
            if pat.search(haystack):
                found.append(label)
        if found:
            messages.append(
                Message(
                    Severity.ERROR,
                    f"found unfilled content: {', '.join(found)}",
                    fix="replace the placeholder with real content or remove it",
                )
            )
            score -= 15 * len(found)

        if skill.word_count > MAX_WORDS:
            messages.append(
                Message(
                    Severity.WARNING,
                    f"body is ~{skill.word_count} words, exceeding {MAX_WORDS}; skills should stay concise — "
                    "move details into references/ or other companion files",
                    fix="move details into references/ and keep SKILL.md an overview",
                )
            )
            score -= 15

        line_count = len(skill.body.splitlines())
        if line_count > MAX_LINES:
            messages.append(
                Message(
                    Severity.WARNING,
                    f"body is {line_count} lines, exceeding {MAX_LINES}; split details into references/",
                    fix="move reference material into companion files and link them",
                )
            )
            score -= 10

        flagged = 0
        for info, content in fences(skill.body):
            if info.strip() or flagged >= 3:
                continue
            if sum(1 for ln in content.splitlines() if CODE_LIKE_RE.match(ln)) >= 2:
                flagged += 1
                messages.append(
                    Message(
                        Severity.WARNING,
                        "fenced code block without a language annotation",
                        fix="add the language after the opening fence, e.g. ```python",
                    )
                )
                score -= 10

        return self._result(score, messages)
