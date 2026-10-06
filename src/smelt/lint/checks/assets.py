"""Asset reference check: do local files referenced in the body actually exist."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote, urlparse

from smelt.lint.checks.base import Check
from smelt.lint.models import CheckResult, Message, Severity, SkillDoc

LOCAL_LINK_RE = re.compile(r"\[[^\]]*\]\(((?!https?://|mailto:|#)[^)\s]+)\)")
BACKTICK_PATH_RE = re.compile(r"`((?:scripts|templates|assets|references|examples|docs|tests)/[^`\s]+)`")


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
        for m in LOCAL_LINK_RE.finditer(skill.body):
            refs.add(m.group(1))
        for m in BACKTICK_PATH_RE.finditer(skill.body):
            refs.add(m.group(1))

        missing = sorted(r for r in refs if not _exists(skill.dir, r))

        messages: list[Message] = []
        score = 100.0
        if missing:
            for r in missing[:5]:
                messages.append(Message(Severity.ERROR, f"referenced file does not exist: {r}"))
            score -= min(60, 20 * len(missing))

        return self._result(score, messages)
