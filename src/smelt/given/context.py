"""given contexts: fixture files, env vars, extra prompts — materialized into an
isolated workspace before the case runs."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ContextSpec:
    """Declarative context description, materialized by the runner.

    - ``files``: (source path → workspace-relative destination) pairs;
      a source may be a file or a directory.
    - ``env``: variables injected into the tool execution environment.
    - ``prompt``: context text appended to the system prompt (background, etc.).
    - ``vars``: free-form variables passed through to agents / assertions.
    """

    files: tuple[tuple[Path, Path], ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)
    prompt: str | None = None
    vars: Mapping[str, Any] = field(default_factory=dict)

    def merge(self, other: ContextSpec) -> ContextSpec:
        return ContextSpec(
            files=self.files + other.files,
            env={**self.env, **other.env},
            prompt="\n\n".join(p for p in (self.prompt, other.prompt) if p) or None,
            vars={**self.vars, **other.vars},
        )


def context(
    *,
    files: str | os.PathLike[str] | Mapping[str, str | os.PathLike[str]] | None = None,
    prompt: str | None = None,
    env: Mapping[str, str] | None = None,
    vars: Mapping[str, Any] | None = None,
) -> ContextSpec:
    """Build a given context.

    Two styles for ``files``:

    - single path: ``context(files="fixtures/repo")`` → contents land at the
      workspace root;
    - mapping: ``context(files={"data/input.csv": "fixtures/input.csv"})`` →
      explicit destinations inside the workspace.
    """
    entries: list[tuple[Path, Path]] = []
    if files is not None:
        if isinstance(files, (str, os.PathLike)):
            entries.append((Path(files), Path(".")))
        else:
            entries.extend((Path(src), Path(dst)) for dst, src in files.items())
    return ContextSpec(
        files=tuple(entries),
        env=dict(env or {}),
        prompt=prompt,
        vars=dict(vars or {}),
    )


@dataclass
class CaseContext:
    """Materialized runtime context; agents access the workspace through it."""

    workspace: Path
    env: dict[str, str] = field(default_factory=dict)
    prompt: str | None = None
    vars: dict[str, Any] = field(default_factory=dict)

    def materialize(self, specs: list[ContextSpec]) -> None:
        for spec in specs:
            for src, dst in spec.files:
                self._copy_into(src, dst)
            self.env.update(spec.env)
            self.vars.update(spec.vars)
            if spec.prompt:
                self.prompt = "\n\n".join(p for p in (self.prompt, spec.prompt) if p)

    def _copy_into(self, src: Path, dst: Path) -> None:
        if not src.exists():
            raise FileNotFoundError(f"context file does not exist: {src}")
        target = self.workspace / dst
        if src.is_dir():
            shutil.copytree(src, target, dirs_exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)


def fresh_workspace(keep_dir: Path | None = None, name: str = "case") -> Path:
    """Create an isolated workspace. Uses the system temp dir when keep_dir is None."""
    if keep_dir is None:
        return Path(tempfile.mkdtemp(prefix=f"smelt-{name}-"))
    keep_dir.mkdir(parents=True, exist_ok=True)
    return keep_dir
