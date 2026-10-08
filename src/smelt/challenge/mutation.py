"""Mutation check: deliberately break the skill, watch whether cases notice.

Mutants are deterministic text transforms applied to an isolated copy of the
skill — the original directory is never touched. A mutant that cannot be
applied (missing section, no matching sentence, unparsable result) is
reported as invalid and excluded from the mutation score.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from smelt.evaluate import _scan_skill_refs

_CONSTRAINT_RE = re.compile(r"(?i)\b(do not|don't|never|must not)\b|不要|禁止|切勿|不得")
_REQUIREMENT_RE = re.compile(r"(?i)\b(must|always|required|be sure to)\b|必须|务必|总是")
_SECTION_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class Mutant:
    id: str  # "drop_section:Boundaries" / "drop_reference:references/x.md" / "drop_constraint"
    kind: str  # drop_section | drop_reference | drop_constraint | drop_requirement
    target: str = ""


def _skill_dir(skill: Path) -> Path:
    return skill.parent if skill.is_file() else skill


def _sections(text: str) -> list[tuple[str, int, int]]:
    """(title, start, end) per '## ' section; end = next '##' start or EOF."""
    matches = list(_SECTION_RE.finditer(text))
    return [
        (m.group(1), m.start(), matches[i + 1].start() if i + 1 < len(matches) else len(text))
        for i, m in enumerate(matches)
    ]


def generate_mutants(skill: Path) -> list[Mutant]:
    """One mutant per '##' section and per referenced file, plus one
    drop_constraint and one drop_requirement."""
    skill_dir = _skill_dir(skill)
    doc = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
    mutants = [
        Mutant(id=f"drop_section:{title}", kind="drop_section", target=title)
        for title, _, _ in _sections(doc)
    ]
    refs = set(_scan_skill_refs(doc))
    ref_root = skill_dir / "references"
    if ref_root.is_dir():
        refs |= {f"references/{p.name}" for p in sorted(ref_root.iterdir()) if p.is_file()}
    mutants += [Mutant(id=f"drop_reference:{r}", kind="drop_reference", target=r) for r in sorted(refs)]
    mutants.append(Mutant(id="drop_constraint", kind="drop_constraint"))
    mutants.append(Mutant(id="drop_requirement", kind="drop_requirement"))
    return mutants


def _valid(skill_dir: Path) -> bool:
    """The mutated copy still parses: SKILL.md exists with non-empty body."""
    md = skill_dir / "SKILL.md"
    if not md.is_file():
        return False
    body = md.read_text(encoding="utf-8")
    if body.startswith("---"):
        parts = body.split("---", 2)
        body = parts[2] if len(parts) == 3 else ""
    return bool(body.strip())


def apply_mutant(skill: Path, mutant: Mutant, dest_root: Path) -> Path | None:
    """Copy the skill into dest_root/<slug>, apply the mutation, return the copy.
    None when the mutant cannot be applied or leaves an invalid skill."""
    src = _skill_dir(skill)
    slug = re.sub(r"[^\w.-]+", "_", mutant.id)
    dest = dest_root / slug
    shutil.copytree(src, dest, dirs_exist_ok=True)
    md = dest / "SKILL.md"
    text = md.read_text(encoding="utf-8")

    if mutant.kind == "drop_section":
        hit = [s for s in _sections(text) if s[0] == mutant.target]
        if not hit:
            shutil.rmtree(dest)
            return None
        _, start, end = hit[0]
        md.write_text(text[:start] + text[end:], encoding="utf-8")
    elif mutant.kind == "drop_reference":
        target = dest / mutant.target
        if not target.is_file():
            shutil.rmtree(dest)
            return None
        target.unlink()
    elif mutant.kind in ("drop_constraint", "drop_requirement"):
        pattern = _CONSTRAINT_RE if mutant.kind == "drop_constraint" else _REQUIREMENT_RE
        kept = [ln for ln in text.splitlines() if not pattern.search(ln)]
        if len(kept) == len(text.splitlines()):
            shutil.rmtree(dest)
            return None
        md.write_text("\n".join(kept) + "\n", encoding="utf-8")
    else:
        shutil.rmtree(dest)
        return None

    if not _valid(dest):
        shutil.rmtree(dest)
        return None
    return dest


# ---------------------------------------------------------------------------
# Kill verdict and case rebinding
# ---------------------------------------------------------------------------

from smelt.case import SmeltCase
from smelt.given.agents.smelt import SmeltAgent
from smelt.given.context import ContextSpec
from smelt.results import CaseResult


def killed(baseline: CaseResult, mutant: CaseResult, *, min_delta: float = 0.05) -> bool:
    """A mutant is killed by a case when the score drop exceeds the noise band
    max(min_delta, 2σ) — σ from the baseline's per-run spread (compare() formula)."""
    band = max(min_delta, 2.0 * baseline.score_std)
    return baseline.score - mutant.score > band


def rebind_for_mutation(
    case: SmeltCase,
    skill_dir: Path,
    doctor_llm,
    tools,
    *,
    source_root: Path | None = None,
) -> SmeltCase | None:
    """Return the case re-pointed at a mutated skill copy; None when the case
    cannot perceive mutations (deterministic backend, custom agent, or
    fragment-bound skill).

    ``source_root`` is the ORIGINAL skill directory. When given, context file
    mounts (reference()/reference_folder()/context(files=...)) that point
    under it are re-pointed at ``skill_dir`` — otherwise a drop_reference
    mutant deletes the file only in the copy while the case still mounts the
    original, making the mutation invisible. A mount whose source the mutant
    removed is dropped: the case then runs without that file, which is
    exactly the change it is supposed to notice."""
    from dataclasses import replace

    from smelt.evaluate import _is_deterministic
    from smelt.given.fragments import SkillSpec

    # NOTE: fallback is None on purpose — determinism is a property of the
    # case's own agent/fragments, never of the doctor stand-in LLM.
    if _is_deterministic(case, None):
        return None
    if case.fragments:
        if any(isinstance(f, SkillSpec) for f in case.fragments):
            return None
        return None  # v1: fragment-assembled agents are not rebound
    if source_root is not None and case.contexts:
        case = replace(case, contexts=_rebind_contexts(case.contexts, source_root, skill_dir))
    agent = case.agent
    if agent is None:
        return case.given(SmeltAgent(skill=str(skill_dir), llm=doctor_llm, tools=tools))
    if isinstance(agent, SmeltAgent):
        return replace(
            case,
            agent=SmeltAgent(
                skill=str(skill_dir),
                llm=agent.llm or doctor_llm,
                tools=agent.tools or tools,
                max_turns=agent.max_turns,
                system_prompt=agent.system_prompt,
            ),
        )
    return None  # custom Agent: opaque, cannot rebind


def _rebind_contexts(
    contexts: tuple[ContextSpec, ...],
    source_root: Path,
    skill_dir: Path,
) -> tuple[ContextSpec, ...]:
    """Re-point file mounts living under the original skill root at the
    (possibly mutated) copy; mounts outside the root are left untouched."""
    from dataclasses import replace

    root = source_root.resolve()
    rebound = []
    for spec in contexts:
        files = []
        for src, dst in spec.files:
            try:
                rel = src.resolve().relative_to(root)
            except ValueError:
                files.append((src, dst))  # unrelated mount: keep as-is
                continue
            remapped = skill_dir / rel
            if remapped.exists():
                files.append((remapped, dst))
            # else: the mutant removed this source — mount nothing, so the
            # case perceives the deletion instead of a FileNotFoundError.
        rebound.append(replace(spec, files=tuple(files)))
    return tuple(rebound)
