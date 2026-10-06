"""Structure check: section organization and body substance."""

from __future__ import annotations

from smelt.lint.checks.base import Check
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
                Message(Severity.ERROR, f"only {len(h2)} H2 section(s); recommend at least {MIN_SECTIONS} (e.g. When to use / Steps / Examples)")
            )
            score -= 40
        elif len(h2) < 3:
            messages.append(Message(Severity.WARNING, "few sections; consider adding examples or caveats"))
            score -= 10

        if skill.word_count < MIN_WORDS:
            messages.append(
                Message(Severity.ERROR, f"body is only ~{skill.word_count} words; too thin for an agent to act on")
            )
            score -= 40
        elif skill.word_count < IDEAL_MIN_WORDS:
            messages.append(Message(Severity.WARNING, f"body is ~{skill.word_count} words; aim for at least {IDEAL_MIN_WORDS}"))
            score -= 10

        return self._result(score, messages)
