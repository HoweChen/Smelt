"""Asset reference check: do local files referenced in the body actually exist."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote, urlparse

from smelt.lint.checks.base import Check
from smelt.lint.markdown import code_contents, link_targets, prose_segments
from smelt.lint.models import CheckResult, Message, Severity, SkillDoc

BACKTICK_PATH_RE = re.compile(r"`((?:scripts|templates|assets|references|examples|docs|tests)/[^`\s]+)`")
# bare prose mentions: "see references/missing.md." — trailing punctuation stripped on use
PROSE_PATH_RE = re.compile(r"(?<![\w/`])((?:scripts|templates|assets|references|examples|docs|tests)/[^\s`\"'\]]+)")

_EXTERNAL_PREFIXES = ("http://", "https://", "mailto:", "#")


def _exists(skill_dir: Path, ref: str) -> bool:
    ref = unquote(urlparse(ref).path)
    if not ref:
        return True
    return (skill_dir / ref).exists()


class AssetCheck(Check):
    check_id = "assets"
    name = "Asset references (local file existence)"
    default_weight = 1.0

    def run(self, skill: SkillDoc) -> CheckResult:
        refs: set[str] = set()
        # markdown links/images: resolved by the CommonMark parser, so the
        # closing ")" of a link target can never leak into the reference
        for target in link_targets(skill.body):
            if not target.startswith(_EXTERNAL_PREFIXES):
                refs.add(target)
        for m in BACKTICK_PATH_RE.finditer(skill.body):
            refs.add(m.group(1))
        # bare prose mentions: scan text segments (never link targets) and
        # code blocks, preserving the old raw-body coverage
        for segment in prose_segments(skill.body) + code_contents(skill.body):
            for m in PROSE_PATH_RE.finditer(segment):
                refs.add(m.group(1).rstrip(".,;:!?"))

        missing = sorted(r for r in refs if not _exists(skill.dir, r))

        messages: list[Message] = []
        score = 100.0
        if missing:
            for r in missing[:5]:
                messages.append(Message(Severity.ERROR, f"referenced file does not exist: {r}"))
            score -= min(60, 20 * len(missing))

        return self._result(score, messages)
