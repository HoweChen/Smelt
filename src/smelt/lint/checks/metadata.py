"""Metadata check: frontmatter, name, description."""

from __future__ import annotations

import re

from smelt.lint.checks.base import Check
from smelt.lint.models import CheckResult, Message, Severity, SkillDoc

DESC_MIN = 30
DESC_MAX = 500
NAME_MAX = 64
NAME_RE = re.compile(r"^[a-z0-9-]+$")
RESERVED_WORDS = ("anthropic", "claude")
NON_THIRD_PERSON_RE = re.compile(r"^(i['\s]|we\s|you\s)", re.IGNORECASE)


class MetadataCheck(Check):
    check_id = "metadata"
    name = "Metadata (frontmatter / name / description)"
    default_weight = 1.5

    def run(self, skill: SkillDoc) -> CheckResult:
        messages: list[Message] = []
        score = 100.0

        if not skill.frontmatter:
            messages.append(
                Message(
                    Severity.ERROR,
                    "missing YAML frontmatter (--- delimited metadata block)",
                    fix="add a --- delimited YAML frontmatter block with name and description",
                )
            )
            return self._result(0.0, messages)

        raw_name = skill.frontmatter.get("name")
        if not raw_name:
            messages.append(
                Message(
                    Severity.ERROR,
                    "frontmatter is missing the name field",
                    fix="add 'name: <skill-name>' to the frontmatter",
                )
            )
            score -= 40
        else:
            name = str(raw_name)
            if name != skill.dir.name:
                messages.append(
                    Message(
                        Severity.WARNING,
                        f"name ({name}) does not match the directory name ({skill.dir.name})",
                        fix="rename the directory or the frontmatter name so they match",
                    )
                )
                score -= 10
            if not NAME_RE.fullmatch(name):
                messages.append(
                    Message(
                        Severity.ERROR,
                        f"name ({name}) must use lowercase letters, numbers and hyphens only",
                        fix="rename to match ^[a-z0-9-]+$, e.g. weekly-report",
                    )
                )
                score -= 10
            if len(name) > NAME_MAX:
                messages.append(
                    Message(
                        Severity.ERROR,
                        f"name is {len(name)} chars, exceeding {NAME_MAX}",
                        fix="shorten the name to 64 characters or fewer",
                    )
                )
                score -= 10
            if any(w in name.lower() for w in RESERVED_WORDS):
                messages.append(
                    Message(
                        Severity.ERROR,
                        "name contains a reserved word (anthropic/claude)",
                        fix="rename without the reserved words anthropic or claude",
                    )
                )
                score -= 10

        desc = skill.description
        if not desc:
            messages.append(
                Message(
                    Severity.ERROR,
                    "frontmatter is missing the description field",
                    fix="add a description stating what the skill does and when to use it",
                )
            )
            score -= 40
        elif len(desc) < DESC_MIN:
            messages.append(
                Message(
                    Severity.WARNING,
                    f"description too short ({len(desc)} chars); agents can't tell when to trigger",
                    fix="expand with what the skill does and concrete trigger scenarios",
                )
            )
            score -= 20
        elif len(desc) > DESC_MAX:
            messages.append(
                Message(
                    Severity.WARNING,
                    f"description too long ({len(desc)} chars)",
                    fix="trim to the essential capability and trigger scenarios",
                )
            )
            score -= 10

        if desc and NON_THIRD_PERSON_RE.match(desc.strip()):
            messages.append(
                Message(
                    Severity.WARNING,
                    "description must be written in third person (no I/we/you openings)",
                    fix="rewrite in third person, e.g. 'Processes Excel files. Use when ...'",
                )
            )
            score -= 10

        return self._result(score, messages)
