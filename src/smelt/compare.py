"""compare(): version-to-version diff of two skill evaluations.

The question compare() answers: "did the new skill version actually get
better?" — not just "did the numbers change". A score difference only counts
when it exceeds the noise band ``max(min_delta, 2σ)``, where σ combines both
sides' per-run spreads (see repeated sampling in runner.py). Independently of
the mean, a pass^k drop (True → False) is flagged as a reliability regression.

Inputs are flexible: SkillEvaluation objects, saved .json report paths, or
raw payload dicts — so two versions evaluated at different times still
compare.

Usage::

    from smelt import compare

    diff = compare("reports/skill-v1.json", "reports/skill-v2.json")
    print(diff.to_markdown())
    diff.assert_no_regression()  # CI gate

CLI::

    smelt compare reports/skill-v1.json reports/skill-v2.json --min-delta 0.05
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

VERDICTS = ("improved", "regressed", "unchanged", "added", "removed")


@dataclass(frozen=True)
class CaseDiff:
    """One case's baseline → candidate comparison."""

    case: str
    verdict: str  # one of VERDICTS
    baseline_score: float | None = None
    candidate_score: float | None = None
    diff: float | None = None
    baseline_std: float = 0.0
    candidate_std: float = 0.0
    baseline_pass_hat: bool | None = None
    candidate_pass_hat: bool | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "case": self.case,
            "verdict": self.verdict,
            "baseline_score": self.baseline_score,
            "candidate_score": self.candidate_score,
            "diff": self.diff,
            "baseline_std": self.baseline_std,
            "candidate_std": self.candidate_std,
            "baseline_pass_hat": self.baseline_pass_hat,
            "candidate_pass_hat": self.candidate_pass_hat,
            "reason": self.reason,
        }


@dataclass
class CompareResult:
    """Whole-report comparison: per-case diffs plus overall movement."""

    baseline_name: str
    candidate_name: str
    cases: list[CaseDiff] = field(default_factory=list)
    min_delta: float = 0.05
    baseline_overall: float | None = None
    candidate_overall: float | None = None
    coverage_changes: list[str] = field(default_factory=list)  # informational reached→unreached transitions

    @property
    def regressions(self) -> list[CaseDiff]:
        return [d for d in self.cases if d.verdict == "regressed"]

    @property
    def improvements(self) -> list[CaseDiff]:
        return [d for d in self.cases if d.verdict == "improved"]

    @property
    def has_regression(self) -> bool:
        return bool(self.regressions)

    @property
    def overall_diff(self) -> float | None:
        if self.baseline_overall is None or self.candidate_overall is None:
            return None
        return self.candidate_overall - self.baseline_overall

    def assert_no_regression(self) -> CompareResult:
        """CI gate: raise AssertionError listing every regressed case."""
        if self.has_regression:
            names = ", ".join(d.case for d in self.regressions)
            raise AssertionError(
                f"regression detected in {self.candidate_name} vs {self.baseline_name}: {names}"
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        counts = {v: sum(1 for d in self.cases if d.verdict == v) for v in VERDICTS}
        return {
            "baseline": {"name": self.baseline_name, "overall": self.baseline_overall},
            "candidate": {"name": self.candidate_name, "overall": self.candidate_overall},
            "overall_diff": self.overall_diff,
            "min_delta": self.min_delta,
            "has_regression": self.has_regression,
            "coverage_changes": self.coverage_changes,
            "counts": counts,
            "cases": [d.to_dict() for d in self.cases],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def to_markdown(self) -> str:
        lines = [f"# Skill Comparison: {self.baseline_name} → {self.candidate_name}", ""]
        if self.overall_diff is not None:
            lines.append(
                f"- Overall: {self.baseline_overall:.1f} → {self.candidate_overall:.1f} "
                f"(**{self.overall_diff:+.1f}**)"
            )
        counts = {v: sum(1 for d in self.cases if d.verdict == v) for v in VERDICTS}
        lines.append(
            "- Cases: "
            + " · ".join(f"{counts[v]} {v}" for v in VERDICTS)
            + f" · noise band = max({self.min_delta:.2f}, 2σ)"
        )
        lines += ["", "| Case | Baseline | Candidate | Δ | Verdict | Notes |", "|---|---|---|---|---|---|"]
        marks = {"improved": "🟢 improved", "regressed": "🔴 regressed", "unchanged": "⚪ unchanged",
                 "added": "🔵 added", "removed": "⚫ removed"}
        for d in self.cases:
            b = f"{d.baseline_score:.2f}" if d.baseline_score is not None else "-"
            c = f"{d.candidate_score:.2f}" if d.candidate_score is not None else "-"
            diff = f"{d.diff:+.2f}" if d.diff is not None else "-"
            lines.append(f"| {d.case} | {b} | {c} | {diff} | {marks[d.verdict]} | {d.reason} |")
        lines.append("")
        if self.coverage_changes:
            lines.append("**Reference coverage changes:** " + "; ".join(self.coverage_changes))
            lines.append("")
        if self.has_regression:
            lines.append(f"**Regressions:** {', '.join(d.case for d in self.regressions)}")
            lines.append("")
        return "\n".join(lines)

    def save(self, path: str | Path) -> Path:
        """Write the report; .json suffix → JSON, otherwise markdown."""
        target = Path(path)
        content = self.to_json() if target.suffix == ".json" else self.to_markdown()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content + "\n", encoding="utf-8")
        return target


def _as_payload(source: Any) -> dict[str, Any]:
    """Normalize an input to the evaluate-report dict shape."""
    from smelt.evaluate import SkillEvaluation

    if isinstance(source, SkillEvaluation):
        return source.to_dict()
    if isinstance(source, dict):
        return source
    if isinstance(source, (str, Path)):
        return json.loads(Path(source).read_text(encoding="utf-8"))
    raise TypeError(
        f"compare() accepts SkillEvaluation / payload dict / .json report path, got: {type(source).__name__}"
    )


def _diff_case(name: str, base: dict | None, cand: dict | None, min_delta: float) -> CaseDiff:
    if base is None:
        return CaseDiff(case=name, verdict="added", candidate_score=cand["score"],
                        candidate_pass_hat=cand.get("pass_hat"))
    if cand is None:
        return CaseDiff(case=name, verdict="removed", baseline_score=base["score"],
                        baseline_pass_hat=base.get("pass_hat"))

    b_score, c_score = base["score"], cand["score"]
    b_std, c_std = base.get("score_std") or 0.0, cand.get("score_std") or 0.0
    b_hat, c_hat = base.get("pass_hat"), cand.get("pass_hat")
    diff = c_score - b_score

    if cand.get("error"):
        return CaseDiff(case=name, verdict="regressed", baseline_score=b_score, candidate_score=c_score,
                        diff=diff, baseline_std=b_std, candidate_std=c_std,
                        baseline_pass_hat=b_hat, candidate_pass_hat=c_hat,
                        reason=f"candidate run error: {cand['error']}")

    common = {"baseline_score": b_score, "candidate_score": c_score, "diff": diff,
              "baseline_std": b_std, "candidate_std": c_std,
              "baseline_pass_hat": b_hat, "candidate_pass_hat": c_hat}
    band = max(min_delta, 2.0 * math.sqrt(b_std**2 + c_std**2))
    if abs(diff) > band:
        return CaseDiff(case=name, verdict="improved" if diff > 0 else "regressed",
                        reason=f"|Δ|={abs(diff):.2f} > band {band:.2f}", **common)
    if b_hat is True and c_hat is False:
        return CaseDiff(case=name, verdict="regressed",
                        reason="reliability drop: pass^k True → False within noise band", **common)
    if b_hat is False and c_hat is True:
        return CaseDiff(case=name, verdict="improved",
                        reason="reliability gain: pass^k False → True within noise band", **common)
    return CaseDiff(case=name, verdict="unchanged",
                    reason=f"|Δ|={abs(diff):.2f} within band {band:.2f}", **common)


def compare(baseline: Any, candidate: Any, *, min_delta: float = 0.05) -> CompareResult:
    """Diff two skill evaluations (SkillEvaluation / payload dict / .json path).

    A case's score change is significant only when |Δ| exceeds
    ``max(min_delta, 2σ)``; a pass^k drop is a reliability regression even
    within the band. ``assert_no_regression()`` on the result is the CI gate.
    """
    if min_delta < 0:
        raise ValueError(f"min_delta must not be negative, got {min_delta}")
    base = _as_payload(baseline)
    cand = _as_payload(candidate)

    base_cases = {c["case"]: c for c in base.get("behavior") or []}
    cand_cases = {c["case"]: c for c in cand.get("behavior") or []}
    names = list(dict.fromkeys([*base_cases, *cand_cases]))  # stable union order

    def _reached(payload: dict[str, Any]) -> dict[str, bool]:
        return {c["path"]: c.get("reached", 0) > 0 for c in (payload.get("reference_coverage") or [])}

    base_reach, cand_reach = _reached(base), _reached(cand)
    changes = []
    if "reference_coverage" in cand:  # pre-feature reports carry no coverage key — never fabricate changes
        changes = [
            f"{p}: reached → unreached"
            for p in base_reach
            if base_reach[p] and not cand_reach.get(p, False)
        ]

    return CompareResult(
        baseline_name=base.get("skill", {}).get("name", "baseline"),
        candidate_name=cand.get("skill", {}).get("name", "candidate"),
        cases=[_diff_case(n, base_cases.get(n), cand_cases.get(n), min_delta) for n in names],
        min_delta=min_delta,
        baseline_overall=base.get("overall", {}).get("score"),
        candidate_overall=cand.get("overall", {}).get("score"),
        coverage_changes=changes,
    )
