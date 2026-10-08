"""@mutate_check: declare which skill part a case guards; doctor verifies the claim.

The decorator only MARKS the function (attribute, not a global registry), so
loading stays module-local and order-free. Doctor's loader calls decorated
zero-arg functions to obtain their cases; `smelt run` / `smelt evaluate`
never execute functions at import (unchanged behavior).
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from smelt.case import SmeltCase
from smelt.challenge.mutation import _sections


@dataclass(frozen=True)
class MutateSpec:
    guards: tuple[str, ...] = ()  # "section:<heading>" | "reference:<path>"


def mutate_check(func: Callable | None = None, *, guards: str | Sequence[str] | None = None):
    """Mark a zero-arg function returning a SmeltCase as a mutation-checked case.

    Bare:        @mutate_check
    With claims: @mutate_check(guards="section:Boundaries")
    """
    spec = MutateSpec(
        guards=() if guards is None else (guards,) if isinstance(guards, str) else tuple(guards)
    )

    def deco(f: Callable) -> Callable:
        f._smelt_mutate = spec  # type: ignore[attr-defined]
        return f

    return deco(func) if callable(func) else deco


def load_cases_and_guards(path: Path) -> tuple[list[SmeltCase], dict[str, MutateSpec]]:
    """Import a case file; collect module-level SmeltCase instances / a cases
    list (same rule as cli._load_cases) plus decorated functions (called to
    obtain their case). Returns (cases, guards-by-case-name)."""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load case file: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cases: list[SmeltCase] = []
    guards: dict[str, MutateSpec] = {}
    explicit = getattr(module, "cases", None)
    if explicit is not None:
        cases.extend(c for c in explicit if isinstance(c, SmeltCase))
    else:
        cases.extend(v for v in vars(module).values() if isinstance(v, SmeltCase))
    for v in vars(module).values():
        mark = getattr(v, "_smelt_mutate", None)
        if mark is None or not callable(v):
            continue
        case = v()
        if not isinstance(case, SmeltCase):
            raise TypeError(f"@mutate_check function {v.__name__} must return a SmeltCase")
        cases.append(case)
        if mark.guards:
            guards[case.name] = mark
    return cases, guards


def find_dangling_guards(guards: dict[str, MutateSpec], skill: Path) -> list[str]:
    """Every guard claim must resolve to an existing '##' section or reference
    file. Returns human-readable dangling entries (empty when all resolve)."""
    skill_dir = skill.parent if skill.is_file() else skill
    doc = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
    titles = {t for t, _, _ in _sections(doc)}
    dangling: list[str] = []
    for case_name, spec in guards.items():
        for g in spec.guards:
            kind, _, target = g.partition(":")
            if kind == "section" and target not in titles:
                dangling.append(f"{case_name}: guards missing section {target!r}")
            elif kind == "reference" and not (skill_dir / target).is_file():
                dangling.append(f"{case_name}: guards missing reference {target!r}")
            elif kind not in ("section", "reference"):
                dangling.append(f"{case_name}: unknown guard kind {kind!r} (expected section:/reference:)")
    return dangling
