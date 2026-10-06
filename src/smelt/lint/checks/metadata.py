"""Metadata check: frontmatter, name, description."""

from __future__ import annotations

from smelt.lint.checks.base import Check
from smelt.lint.models import CheckResult, Message, Severity, SkillDoc

DESC_MIN = 30
DESC_MAX = 500


class MetadataCheck(Check):
    check_id = "metadata"
    name = "Metadata (frontmatter / name / description)"
    default_weight = 1.5

    def run(self, skill: SkillDoc) -> CheckResult:
        messages: list[Message] = []
        score = 100.0

        if not skill.frontmatter:
            messages.append(Message(Severity.ERROR, "missing YAML frontmatter (--- delimited metadata block)"))
            return self._result(0.0, messages)

        if not skill.frontmatter.get("name"):
            messages.append(Message(Severity.ERROR, "frontmatter is missing the name field"))
            score -= 40
        elif skill.frontmatter["name"] != skill.dir.name:
            messages.append(
                Message(
                    Severity.WARNING,
                    f"name ({skill.frontmatter['name']}) does not match the directory name ({skill.dir.name})",
                )
            )
            score -= 10

        desc = skill.description
        if not desc:
            messages.append(Message(Severity.ERROR, "frontmatter is missing the description field"))
            score -= 40
        elif len(desc) < DESC_MIN:
            messages.append(
                Message(Severity.WARNING, f"description too short ({len(desc)} chars); agents can't tell when to trigger")
            )
            score -= 20
        elif len(desc) > DESC_MAX:
            messages.append(Message(Severity.WARNING, f"description too long ({len(desc)} chars)"))
            score -= 10

        return self._result(score, messages)
