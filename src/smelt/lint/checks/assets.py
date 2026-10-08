"""Asset reference check: do local files referenced in the body actually exist."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote, urlparse

from smelt.lint.checks.base import Check
from smelt.lint.markdown import code_contents, headings, inline_code, link_targets, prose_segments
from smelt.lint.models import CheckResult, Message, Severity, SkillDoc

BACKTICK_PATH_RE = re.compile(r"`((?:scripts|templates|assets|references|examples|docs|tests)/[^`\s]+)`")
# bare prose mentions: "see references/missing.md." — trailing punctuation stripped on use
PROSE_PATH_RE = re.compile(r"(?<![\w/`])((?:scripts|templates|assets|references|examples|docs|tests)/[^\s`\"'\]]+)")
BACKSLASH_DIR_RE = re.compile(r"^(?:scripts|templates|assets|references|examples|docs|tests)\\")
_SLUG_STRIP_RE = re.compile(r"[^\w\s-]")

_EXTERNAL_PREFIXES = ("http://", "https://", "mailto:", "#")


def _heading_slugs(path: Path) -> set[str]:
    """GitHub-style slugs of a markdown file's headings (duplicates get -1, -2)."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return set()
    seen: dict[str, int] = {}
    slugs: set[str] = set()
    for _, title in headings(text):
        base = _SLUG_STRIP_RE.sub("", title).strip().lower().replace(" ", "-")
        n = seen.get(base, 0)
        seen[base] = n + 1
        slugs.add(base if n == 0 else f"{base}-{n}")
    return slugs


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
                messages.append(
                    Message(
                        Severity.ERROR,
                        f"referenced file does not exist: {r}",
                        fix="create the file or remove the reference",
                    )
                )
            score -= min(60, 20 * len(missing))

        local_targets = [t for t in link_targets(skill.body) if not t.startswith(_EXTERNAL_PREFIXES)]

        # note: markdown-it-py URL-normalizes hrefs, so a backslash arrives as
        # %5C — unquote before testing
        backslash_hits = [t for t in local_targets if "\\" in unquote(t)]
        backslash_hits += [s for s in inline_code(skill.body) if BACKSLASH_DIR_RE.match(s)]
        for hit in backslash_hits[:3]:
            messages.append(
                Message(
                    Severity.WARNING,
                    f"Windows-style path with backslashes: {hit}; use forward slashes",
                    fix="use forward slashes in all paths",
                )
            )
            score -= 10

        anchored = 0
        for t in local_targets:
            path_part, _, frag = t.partition("#")
            if not frag or not path_part.lower().endswith(".md"):
                continue
            target = skill.dir / unquote(urlparse(path_part).path)
            if not target.is_file():
                continue  # missing file is already reported by the existence check
            frag = unquote(frag)
            if frag not in _heading_slugs(target):
                anchored += 1
                if anchored <= 3:  # messages and deductions share the cap
                    messages.append(
                        Message(
                            Severity.WARNING,
                            f"anchor '#{frag}' not found in {path_part}",
                            fix="point the link at an existing heading or fix the anchor",
                        )
                    )
                    score -= 10

        nested = 0
        for t in local_targets:
            path_part = unquote(urlparse(t.partition("#")[0]).path)
            if not path_part.lower().endswith(".md"):
                continue
            target = skill.dir / path_part
            if not target.is_file():
                continue
            try:
                nested_links = link_targets(target.read_text(encoding="utf-8"), include_images=False)
            except OSError:
                continue
            if any(not n.startswith(_EXTERNAL_PREFIXES) for n in nested_links):
                nested += 1
                if nested <= 3:  # messages and deductions share the cap
                    messages.append(
                        Message(
                            Severity.WARNING,
                            f"{path_part} links to further local files; keep references one level deep",
                            fix="link the nested files directly from SKILL.md",
                        )
                    )
                    score -= 10

        return self._result(score, messages)
