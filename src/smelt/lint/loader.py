"""Load and parse SKILL.md files."""

from __future__ import annotations

import re
from pathlib import Path

from smelt.lint.models import SkillDoc

FRONTMATTER_RE = re.compile(r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*\r?\n?", re.DOTALL)
HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*$", re.MULTILINE)


class SkillLoadError(Exception):
    """A SKILL.md could not be loaded or parsed."""


def _parse_simple_yaml(text: str) -> dict:
    """Minimal dependency-free YAML subset parser (scalars, inline and block lists).

    PyYAML is preferred when available; this implementation only covers the
    frontmatter idioms common in SKILL.md. Install pyyaml for complex nesting.
    """

    data: dict = {}
    current_key: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("- ") and current_key:
            if not isinstance(data.get(current_key), list):
                data[current_key] = []
            data[current_key].append(stripped[2:].strip().strip("'\""))
            continue
        if ":" not in stripped:
            continue
        key, _, value = stripped.partition(":")
        key = key.strip()
        value = value.strip()
        current_key = key
        if value.startswith("[") and value.endswith("]"):
            data[key] = [v.strip().strip("'\"") for v in value[1:-1].split(",") if v.strip()]
        elif value:
            data[key] = value.strip("'\"")
        else:
            data[key] = None
    return data


def parse_frontmatter(raw: str) -> tuple[dict, str]:
    """Split raw text into (frontmatter dict, body). Returns ({}, raw) without frontmatter."""
    m = FRONTMATTER_RE.match(raw)
    if not m:
        return {}, raw
    fm_text = m.group(1)
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(fm_text) or {}
        if not isinstance(data, dict):
            data = {}
    except ImportError:
        data = _parse_simple_yaml(fm_text)
    return data, raw[m.end():]


CJK_RE = re.compile(r"[぀-ヿ㐀-鿿豈-﫿]")
LATIN_WORD_RE = re.compile(r"[A-Za-z0-9_'-]+")


def count_words(text: str) -> int:
    """Word count for mixed CJK/Latin text: CJK chars count individually, Latin runs count as words."""
    no_cjk = CJK_RE.sub(" ", text)
    return len(CJK_RE.findall(text)) + len(LATIN_WORD_RE.findall(no_cjk))


def load_skill(path: str | Path) -> SkillDoc:
    """Load a skill directory (or a SKILL.md file path)."""
    p = Path(path)
    skill_md = p / "SKILL.md" if p.is_dir() else p
    if not skill_md.is_file():
        raise SkillLoadError(f"SKILL.md not found: {skill_md}")
    raw = skill_md.read_text(encoding="utf-8")
    frontmatter, body = parse_frontmatter(raw)
    sections = [(len(m.group(1)), m.group(2).strip()) for m in HEADING_RE.finditer(body)]
    word_count = count_words(body)
    name = str(frontmatter.get("name") or skill_md.parent.name)
    description = str(frontmatter.get("description") or "")
    return SkillDoc(
        path=skill_md,
        name=name,
        description=description,
        frontmatter=frontmatter,
        body=body,
        sections=sections,
        word_count=word_count,
    )


def discover_skills(path: str | Path) -> list[Path]:
    """Discover skills: return [path] if it is a skill directory, else scan its children."""
    p = Path(path)
    if (p / "SKILL.md").is_file():
        return [p]
    if not p.is_dir():
        raise SkillLoadError(f"path does not exist or is not a directory: {p}")
    skills = [d for d in sorted(p.iterdir()) if d.is_dir() and (d / "SKILL.md").is_file()]
    if not skills:
        raise SkillLoadError(f"no skill directories containing SKILL.md found under {p}")
    return skills
