"""given fragments for skill resources: reference() / reference_folder().

Both mount files into the case workspace AND register the workspace-relative
paths into CaseContext.references — declaration doubles as registration,
which is what restraint checks and the evaluate coverage report build on.
"""

from __future__ import annotations

import os
from pathlib import Path

from smelt.given.context import ContextSpec

RESOURCE_DIRS = ("references", "scripts", "assets")


def _skill_relative(file: Path) -> Path:
    """Path relative to the enclosing skill root (dir containing SKILL.md);
    falls back to the bare file name."""
    for parent in (file.parent, *file.parent.parents):
        if (parent / "SKILL.md").exists():
            return file.relative_to(parent)
    return Path(file.name)


def reference(path: str | os.PathLike[str]) -> ContextSpec:
    """given(reference("skills/x/references/a.md")) — mount one file at its
    skill-relative workspace path and register it."""
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"reference does not exist: {src}")
    dest = _skill_relative(src)
    return ContextSpec(files=((src, dest),), references=(dest.as_posix(),))


def reference_folder(directory: str | os.PathLike[str]) -> ContextSpec:
    """given(reference_folder("skills/x")) — mount all of the skill's resource
    dirs (references/ scripts/ assets/, whichever exist); a bare directory is
    mounted recursively under workspace/<dir.name>/. Every mounted file is
    registered."""
    d = Path(directory)
    if not d.exists():
        raise FileNotFoundError(f"reference_folder does not exist: {d}")
    entries: list[tuple[Path, Path]] = []
    if (d / "SKILL.md").exists():
        entries.extend((d / s, Path(s)) for s in RESOURCE_DIRS if (d / s).is_dir())
    else:
        entries.append((d, Path(d.name)))
    registered: list[str] = []
    for src, dst in entries:
        for f in sorted(src.rglob("*")):
            if f.is_file():
                registered.append((dst / f.relative_to(src)).as_posix())
    return ContextSpec(files=tuple(entries), references=tuple(registered))
