"""smelt doctor: health check for the yardstick — case suite (mutation check)
and judge (canary). Exit-code mapping lives in the CLI; here ok is a property.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from smelt.case import SmeltCase
from smelt.challenge.canary import CanaryResult, run_canary
from smelt.challenge.guards import find_dangling_guards, load_cases_and_guards
from smelt.challenge.mutation import (
    Mutant,
    apply_mutant,
    generate_mutants,
    killed,
    rebind_for_mutation,
)
from smelt.given.agents.llm import LLMClient
from smelt.results import CaseResult


@dataclass(frozen=True)
class MutantResult:
    mutant: Mutant
    verdict: str  # killed | survived | invalid
    killed_by: tuple[str, ...] = ()
    note: str = ""


@dataclass
class DoctorReport:
    skill_path: str
    cases_tested: int
    mutants: list[MutantResult] = field(default_factory=list)
    mutation_score: float | None = None  # None = N/A (deterministic backend)
    false_guards: list[str] = field(default_factory=list)
    dangling_guards: list[str] = field(default_factory=list)
    red_baselines: list[str] = field(default_factory=list)  # cases failing against the UNMUTATED skill
    canary: CanaryResult | None = None
    notes: list[str] = field(default_factory=list)
    min_score: float = 0.8
    token_usage: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        if self.false_guards or self.dangling_guards or self.red_baselines:
            return False
        if self.canary is not None and not self.canary.calibrated:
            return False
        return not (self.mutation_score is not None and self.mutation_score < self.min_score)

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill": self.skill_path,
            "ok": self.ok,
            "cases_tested": self.cases_tested,
            "mutation_score": self.mutation_score,
            "min_score": self.min_score,
            "mutants": [
                {"id": m.mutant.id, "verdict": m.verdict, "killed_by": list(m.killed_by), "note": m.note}
                for m in self.mutants
            ],
            "false_guards": self.false_guards,
            "dangling_guards": self.dangling_guards,
            "red_baselines": self.red_baselines,
            "canary": (
                {"score": self.canary.score, "calibrated": self.canary.calibrated, "reason": self.canary.reason}
                if self.canary else None
            ),
            "notes": self.notes,
            "token_usage": self.token_usage,
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def to_markdown(self) -> str:
        lines = [f"# smelt doctor: {self.skill_path}", ""]
        lines.append(f"Cases tested: {self.cases_tested}")
        if self.mutation_score is None:
            lines.append("Mutation check: N/A")
        else:
            lines.append(f"Mutation score: {self.mutation_score:.0%} (gate {self.min_score:.0%})")
        lines += ["", "| Mutant | Verdict | Killed by |", "|---|---|---|"]
        for m in self.mutants:
            lines.append(f"| {m.mutant.id} | {m.verdict} | {', '.join(m.killed_by) or '-'} |")
        for m in self.mutants:
            if m.verdict == "survived":
                lines.append(f"\n⚠ surviving mutant `{m.mutant.id}` — {m.note}")
        for g in self.false_guards:
            lines.append(f"\n⚠ false guard: {g}")
        for g in self.dangling_guards:
            lines.append(f"\n⚠ dangling guard: {g}")
        for r in self.red_baselines:
            lines.append(f"\n⚠ red baseline: {r} — fails against the unmutated skill; "
                         "fix the skill (or the case) first, mutation verdicts ignore it")
        if self.canary is not None:
            state = "calibrated ✔" if self.canary.calibrated else "MISCALIBRATED ✘"
            lines.append(f"\n## Judge canary: {state} (score {self.canary.score:.2f})")
            lines.append(f"> {self.canary.reason}")
        for n in self.notes:
            lines.append(f"\n- note: {n}")
        lines.append(f"\n**doctor: {'ok' if self.ok else 'issues found'}**")
        return "\n".join(lines)

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        content = self.to_json() if target.suffix == ".json" else self.to_markdown()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content + "\n", encoding="utf-8")
        return target


def _default_run(cases: list[SmeltCase], skill_dir: Path, times: int | None = None) -> list[CaseResult]:
    """Baseline runs honor the case's own repeat count (supplies σ for the
    kill band); mutant runs pass times=1 (coarse screen)."""
    return [c.run(times=times if times is not None else c.times) for c in cases]


def doctor(
    case_files,
    *,
    skill,
    doctor: LLMClient | None = None,
    min_score: float = 0.8,
    tools=(),
    _run: Callable[[list[SmeltCase], Path, int | None], list[CaseResult]] | None = None,
) -> DoctorReport:
    """Health-check a case suite (mutation) and the judge (canary).

    ``_run`` is the case-running seam: ``_run(cases, skill_dir, times)`` —
    baseline calls pass times=None (honor each case's repeat count, supplying
    σ for the kill band); mutant calls pass times=1. Tests inject canned
    scores through it.
    """
    from smelt.llm_config import LLMConfig

    if doctor is None:
        doctor = LLMConfig.from_role("doctor").build()  # LLMConfigError → caller maps to exit 2
    run = _run or _default_run
    skill_path = Path(skill)
    report = DoctorReport(skill_path=str(skill_path), cases_tested=0, min_score=min_score)

    all_cases: list[SmeltCase] = []
    guards: dict[str, Any] = {}
    for f in case_files:
        cases, g = load_cases_and_guards(Path(f))
        all_cases.extend(cases)
        guards.update(g)
    report.cases_tested = len(all_cases)
    report.dangling_guards = find_dangling_guards(guards, skill_path)

    # -- mutation check -------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        pristine = tmp_path / "pristine"
        shutil.copytree(skill_path if skill_path.is_dir() else skill_path.parent, pristine)

        rebound: list[tuple[SmeltCase, SmeltCase]] = []
        skipped = 0
        for c in all_cases:
            r = rebind_for_mutation(c, pristine, doctor, tools)
            if r is None:
                skipped += 1
            else:
                rebound.append((c, r))
        if not rebound:
            report.mutation_score = None
            report.notes.append(
                f"mutation check N/A: all {skipped} case(s) use a deterministic backend "
                "(scripted traces never re-read the skill)"
            )
        else:
            if skipped:
                report.notes.append(f"{skipped} case(s) skipped (deterministic backend or custom agent)")
            baselines = {c.name: res for (c, r), res in zip(rebound, run([r for _, r in rebound], pristine, None))}

            # Classic mutation testing requires a GREEN baseline: a case that
            # already fails against the unmutated skill cannot prove anything
            # about mutations — exclude it from kill attribution and flag it.
            green = [(orig, r) for (orig, r) in rebound if baselines[orig.name].passed]
            report.red_baselines = [orig.name for (orig, _) in rebound if not baselines[orig.name].passed]
            for name in report.red_baselines:
                report.notes.append(
                    f"case '{name}' fails at baseline ({baselines[name].score:.2f}) — "
                    "fix the skill (or the case) first; mutation verdicts ignore it"
                )

            needed_cells: set[tuple[str, str]] = set()  # (mutant_id, case_name) for guard attribution
            green_names = {orig.name for orig, _ in green}
            for case_name, spec in guards.items():
                if case_name not in green_names:
                    if case_name in baselines:
                        report.notes.append(
                            f"guard(s) on '{case_name}' unverifiable: baseline is red"
                        )
                    continue
                for g in spec.guards:
                    needed_cells.add((f"drop_{g}", case_name))

            if not green:
                report.mutation_score = None
                report.notes.append("mutation check N/A: every case fails at baseline")
                mutants = []
            else:
                mutants = generate_mutants(pristine)
            results: list[MutantResult] = []
            for mutant in mutants:
                dest = apply_mutant(pristine, mutant, tmp_path / "mutants")
                if dest is None:
                    results.append(MutantResult(mutant=mutant, verdict="invalid", note="mutant not applicable"))
                    continue
                killed_by: list[str] = []
                cell_cache: dict[str, CaseResult] = {}
                for orig, _ in green:
                    if killed_by and (mutant.id, orig.name) not in needed_cells:
                        continue  # suite-level short-circuit; cell not needed for attribution
                    mutant_case = rebind_for_mutation(orig, dest, doctor, tools)
                    if mutant_case is None:
                        continue
                    res = run([mutant_case], dest, 1)[0]
                    cell_cache[orig.name] = res
                    if killed(baselines[orig.name], res):
                        killed_by.append(orig.name)
                verdict = "killed" if killed_by else "survived"
                note = "" if verdict == "killed" else (
                    f"no case noticed this change — add a case asserting the behavior "
                    f"touched by {mutant.id}"
                )
                results.append(MutantResult(mutant=mutant, verdict=verdict,
                                            killed_by=tuple(killed_by), note=note))
                # guard attribution: a guarded case must kill its declared mutant itself
                for orig, _ in green:
                    if (
                        (mutant.id, orig.name) in needed_cells
                        and orig.name not in killed_by
                        and (orig.name in cell_cache or killed_by)
                    ):
                        report.false_guards.append(
                            f"{orig.name} claims {mutant.id!r} but did not kill it"
                        )
            scorable = [m for m in results if m.verdict != "invalid"]
            killed_n = sum(1 for m in scorable if m.verdict == "killed")
            report.mutants = results
            report.mutation_score = killed_n / len(scorable) if scorable else None

    # -- judge canary ----------------------------------------------------------
    from smelt.then import LLMJudgeExpectation

    if all_cases and not any(
        isinstance(e, LLMJudgeExpectation) for c in all_cases for e in c.expectations
    ):
        report.notes.append("canary is advisory: loaded cases use no llm_judge assertions")
    report.canary = run_canary(doctor)
    return report
