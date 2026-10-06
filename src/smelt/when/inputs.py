"""when triggers: a sentence, or a directory of files."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TextInput:
    """A one-sentence input, delivered as the user message."""

    content: str

    def render(self, workspace: Path) -> str:
        return self.content


@dataclass(frozen=True)
class DirectoryInput:
    """A directory input: materialized into the workspace, with a file listing
    rendered into the user message."""

    path: Path
    note: str = "Please review the following files in the current workspace"

    def render(self, workspace: Path) -> str:
        if not self.path.exists():
            raise FileNotFoundError(f"when directory does not exist: {self.path}")
        shutil.copytree(self.path, workspace, dirs_exist_ok=True)
        listing = "\n".join(
            str(p.relative_to(workspace)) for p in sorted(workspace.rglob("*")) if p.is_file()
        )
        return f"{self.note}:\n{listing or '(empty directory)'}"


CaseInput = TextInput | DirectoryInput


def text(content: str) -> TextInput:
    """when(text("commit my changes")) — a sentence as the trigger."""
    return TextInput(content=content)


def directory(path: str | os.PathLike[str], *, note: str | None = None) -> DirectoryInput:
    """when(directory("fixtures/messy_repo")) — a file directory as the trigger."""
    kwargs = {"note": note} if note is not None else {}
    return DirectoryInput(path=Path(path), **kwargs)
