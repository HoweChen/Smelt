"""Structure check: section organization and body substance."""

from __future__ import annotations

from itertools import pairwise

from smelt.lint.checks.base import Check
from smelt.lint.markdown import headings
from smelt.lint.models import CheckResult, Message, Severity, SkillDoc

MIN_SECTIONS = 2
MIN_WORDS = 100
IDEAL_MIN_WORDS = 250


class StructureCheck(Check):
    check_id = "structure"
    name = "Document structure (sections / substance)"
    default_weight = 1.2

    def run(self, skill: SkillDoc) -> CheckResult:
        messages: list[Message] = []
        score = 100.0

        h2 = [t for lvl, t in skill.sections if lvl == 2]
        if len(h2) < MIN_SECTIONS:
            messages.append(
                Message(
                    Severity.ERROR,
                    f"only {len(h2)} H2 section(s); recommend at least {MIN_SECTIONS} (e.g. When to use / Steps / Examples)",
                    fix="split the body into sections such as When to use / Steps / Examples",
                )
            )
            score -= 40
        elif len(h2) < 3:
            messages.append(
                Message(
                    Severity.WARNING,
                    "few sections; consider adding examples or caveats",
                    fix="add an Examples or Caveats section",
                )
            )
            score -= 10

        if skill.word_count < MIN_WORDS:
            messages.append(
                Message(
                    Severity.ERROR,
                    f"body is only ~{skill.word_count} words; too thin for an agent to act on",
                    fix="add concrete steps, examples, and failure-path notes",
                )
            )
            score -= 40
        elif skill.word_count < IDEAL_MIN_WORDS:
            messages.append(
                Message(
                    Severity.WARNING,
                    f"body is ~{skill.word_count} words; aim for at least {IDEAL_MIN_WORDS}",
                    fix="expand thin sections with concrete detail",
                )
            )
            score -= 10

        heads = headings(skill.body)
        if sum(1 for lvl, _ in heads if lvl == 1) > 1:
            messages.append(
                Message(
                    Severity.WARNING,
                    "multiple H1 headings; use one H1 as the document title",
                    fix="demote extra H1 headings to H2",
                )
            )
            score -= 10
        levels = [lvl for lvl, _ in heads]
        for a, b in [(a, b) for a, b in pairwise(levels) if b > a + 1][:3]:
            messages.append(
                Message(
                    Severity.WARNING,
                    f"heading level skips from H{a} to H{b}",
                    fix="introduce the intermediate heading level(s)",
                )
            )
            score -= 10

        return self._result(score, messages)
