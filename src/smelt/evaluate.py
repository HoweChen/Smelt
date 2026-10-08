"""Skill evaluation: behavior score (then results) + writing score (LLM review of
SKILL.md) + static lint score, weighted into a report.

Usage::

    from smelt import evaluate_skill

    report = (
        evaluate_skill("skills/commit", judge=judge_llm)
        .with_cases(case_a, case_b)            # behavior cases (unbound cases auto-bind the skill under review)
        .with_lint()                           # static lint score
        .with_writing()                        # LLM writing review (per dimension)
        .with_suggestions(max_items=5)         # LLM improvement suggestions
        .with_weights(behavior=0.5, writing=0.3, lint=0.2)
        .run()
    )
    report.save("reports/commit.md")           # extension picks the format; .to_json() also works
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from smelt.case import SmeltCase
from smelt.given.agents.fixed import FixedAgent
from smelt.given.agents.llm import LLMClient, ScriptedLLM
from smelt.given.agents.smelt import SmeltAgent, smelt_agent
from smelt.given.fragments import LLMSpec
from smelt.lint.checks import run_checks
from smelt.lint.loader import load_skill
from smelt.lint.models import Severity
from smelt.lint.scorer import build_report, grade_of
from smelt.results import CaseResult
from smelt.then.expectations import _extract_json
from smelt.tools import Tool

# Default repeat count for stochastic agents (real LLMs). Deterministic
# backends (fixed_agent / ScriptedLLM) auto-degrade to a single run.
DEFAULT_TIMES = 3

_REF_SCAN_RE = re.compile(
    r"(?:\]\(|`|(?<![\w/]))((?:\./)?(?:references|scripts|assets)/[^\s`)\"'\]]+)"
)


def _scan_skill_refs(document: str) -> list[str]:
    """Referenced resource paths in markdown links, inline code spans, or bare
    prose; normalized (./ stripped, trailing punctuation dropped)."""
    from smelt.refpath import normalize_ref_path

    return sorted({normalize_ref_path(m).rstrip(".,;:!?") for m in _REF_SCAN_RE.findall(document)})

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WritingDimension:
    """One dimension of the writing review."""

    name: str
    score: float  # 0~1
    comment: str = ""


@dataclass(frozen=True)
class WritingAssessment:
    """LLM assessment of the SKILL.md's writing quality."""

    dimensions: tuple[WritingDimension, ...]
    overall_comment: str = ""
    error: str | None = None  # failure reason; dimensions empty when set

    @property
    def score(self) -> float | None:
        if not self.dimensions:
            return None
        return sum(d.score for d in self.dimensions) / len(self.dimensions)


@dataclass
class SkillEvaluation:
    """A comprehensive evaluation report for one skill."""

    skill_path: str
    skill_name: str
    behavior_results: list[CaseResult] = field(default_factory=list)
    lint_report: Any = None  # smelt.lint.models.SkillReport
    writing: WritingAssessment | None = None
    suggestions: list[str] = field(default_factory=list)
    suggestions_error: str | None = None
    suggestions_ran: bool = False  # True when suggestion generation executed (even if empty)
    weights: dict[str, float] = field(default_factory=dict)
    reference_coverage: list[dict[str, Any]] | None = None  # None when the skill references nothing

    # -- per-part scores (0~1; None when disabled or no data) -----------------
    @property
    def behavior_score(self) -> float | None:
        if not self.behavior_results:
            return None
        return sum(r.score for r in self.behavior_results) / len(self.behavior_results)

    @property
    def lint_score(self) -> float | None:
        if self.lint_report is None:
            return None
        return self.lint_report.total_score / 100.0

    @property
    def writing_score(self) -> float | None:
        return self.writing.score if self.writing else None

    @property
    def overall_score(self) -> float | None:
        """Weighted total (0~100). Only parts with data count; weights normalize."""
        parts = {
            "behavior": self.behavior_score,
            "writing": self.writing_score,
            "lint": self.lint_score,
        }
        earned = sum(self.weights.get(k, 0.0) * v for k, v in parts.items() if v is not None)
        total_weight = sum(self.weights.get(k, 0.0) for k, v in parts.items() if v is not None)
        if total_weight == 0:
            return None
        return earned / total_weight * 100.0

    @property
    def grade(self) -> str:
        score = self.overall_score
        return grade_of(score) if score is not None else "-"

    # -- output ---------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "skill": {"path": self.skill_path, "name": self.skill_name},
            "overall": {"score": self.overall_score, "grade": self.grade, "weights": self.weights},
            "scores": {
                "behavior": self.behavior_score,
                "writing": self.writing_score,
                "lint": self.lint_score,
            },
            "behavior": [
                {
                    "case": r.case_name,
                    "score": r.score,
                    "passed": r.passed,
                    "error": r.error,
                    "runs": r.runs,
                    "run_scores": r.run_scores,
                    "run_passed": r.run_passed,
                    "pass_hat": r.pass_hat,
                    "score_std": r.score_std,
                    "expectations": [
                        {
                            "name": e.name,
                            "score": e.score,
                            "threshold": e.threshold,
                            "passed": e.passed,
                            "message": e.message,
                            "runs": e.runs,
                            "score_std": e.score_std,
                        }
                        for e in r.expectations
                    ],
                }
                for r in self.behavior_results
            ],
            "lint": (
                {
                    "total_score": self.lint_report.total_score,
                    "grade": self.lint_report.grade,
                    "checks": [
                        {
                            "check_id": c.check_id,
                            "name": c.name,
                            "score": c.score,
                            "passed": c.passed,
                            "messages": [
                                {"severity": m.severity.value, "text": m.text, **({"fix": m.fix} if m.fix else {})}
                                for m in c.messages
                            ],
                        }
                        for c in self.lint_report.results
                    ],
                }
                if self.lint_report
                else None
            ),
            "writing": (
                {
                    "score": self.writing.score,
                    "overall_comment": self.writing.overall_comment,
                    "error": self.writing.error,
                    "dimensions": [
                        {"name": d.name, "score": d.score, "comment": d.comment}
                        for d in self.writing.dimensions
                    ],
                }
                if self.writing
                else None
            ),
            "suggestions": self.suggestions,
            "suggestions_error": self.suggestions_error,
            "reference_coverage": self.reference_coverage,
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def to_markdown(self) -> str:
        lines = [f"# Skill Evaluation Report: {self.skill_name}", ""]
        score = self.overall_score
        lines += [
            f"- Path: `{self.skill_path}`",
            f"- Overall: **{f'{score:.1f}' if score is not None else '-'} / 100**  Grade: **{self.grade}**",
            "- Parts: " + ", ".join(
                f"{label} {f'{v * 100:.1f}' if v is not None else '-'}"
                for label, v in (
                    ("behavior", self.behavior_score),
                    ("writing", self.writing_score),
                    ("lint", self.lint_score),
                )
            ),
            "",
        ]

        if self.behavior_results:
            lines += ["## Behavior Tests", ""]
            for r in self.behavior_results:
                mark = "✅" if r.passed else "❌"
                spread = f" · n={r.runs} ±{r.score_std * 100:.0f}" if r.runs > 1 else ""
                if r.pass_hat is not None:
                    spread += f" · pass^{r.runs} {'✔' if r.pass_hat else '✘'}"
                lines.append(f"### {mark} {r.case_name} ({r.score * 100:.0f} pts{spread})")
                if r.error:
                    lines.append(f"- Runtime error: {r.error}")
                for err in r.run_errors:
                    lines.append(f"- Run error (counted as 0): {err}")
                for e in r.expectations:
                    icon = "✔" if e.passed else "✘"
                    msg = f" — {e.message}" if e.message else ""
                    e_spread = f" ±{e.score_std:.2f}" if e.runs > 1 else ""
                    lines.append(f"- {icon} `{e.name}` {e.score:.2f}{e_spread} (threshold {e.threshold:.2f}){msg}")
                lines.append("")

        if self.lint_report is not None:
            lines += [
                "## Static Lint",
                "",
                f"Total {self.lint_report.total_score:.1f}, grade {self.lint_report.grade}",
                "",
                "| Check | Score | Result |",
                "|---|---|---|",
            ]
            for c in self.lint_report.results:
                lines.append(f"| {c.name} | {c.score:.1f} | {'✅' if c.passed else '❌'} |")
            lines.append("")

        if self.writing is not None:
            lines += ["## Writing Review (LLM)", ""]
            if self.writing.error:
                lines.append(f"Review failed: {self.writing.error}")
            else:
                lines += ["| Dimension | Score | Comment |", "|---|---|---|"]
                for d in self.writing.dimensions:
                    lines.append(f"| {d.name} | {d.score:.2f} | {d.comment} |")
                if self.writing.overall_comment:
                    lines += ["", f"> {self.writing.overall_comment}"]
            lines.append("")

        if self.reference_coverage:
            lines += ["## Reference Coverage", "", "| Reference | Reached | Origin |", "|---|---|---|"]
            for c in self.reference_coverage:
                reached = f"{c['reached']}×" if c["reached"] else "unreached"
                lines.append(f"| {c['path']} | {reached} | {c['origin']} |")
            lines.append("")

        lines += ["## Improvement Suggestions", ""]
        if self.suggestions:
            lines += [f"{i}. {s}" for i, s in enumerate(self.suggestions, 1)]
        elif self.suggestions_error:
            lines.append(f"Suggestion generation failed: {self.suggestions_error}")
        elif self.suggestions_ran:
            lines.append("No lint issues; nothing to suggest.")
        else:
            lines.append("(not enabled)")
        lines.append("")
        return "\n".join(lines)

    def save(self, path: str | Path, *, format: str | None = None) -> Path:
        """Write the report to a file. format defaults from the extension (.json → json,
        otherwise markdown)."""
        target = Path(path)
        fmt = format or ("json" if target.suffix == ".json" else "markdown")
        content = self.to_json() if fmt == "json" else self.to_markdown()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content + "\n", encoding="utf-8")
        return target


# ---------------------------------------------------------------------------
# LLM judging
# ---------------------------------------------------------------------------

DEFAULT_DIMENSIONS: tuple[str, ...] = (
    "Metadata & naming (does name/description accurately capture the purpose)",
    "Trigger guidance (can an agent tell when to load this skill)",
    "Structure & readability (sectioning, length control)",
    "Examples & edge cases (concrete examples, failure-path notes)",
    "Actionability (steps are explicit, executable, unambiguous)",
)

WRITING_PROMPT = """You are a skill-documentation reviewer. Review the writing quality of the SKILL.md below,
dimension by dimension.

## Document under review (path: {path})
```markdown
{document}
```

## Dimensions (score each 0.0~1.0 with a one-sentence comment)
{dimension_list}

Output JSON only: {{"dimensions": [{{"name": "<dimension>", "score": <0~1>, "comment": "<one sentence>"}}], "overall_comment": "<one sentence overall>"}}"""

SUGGESTIONS_PROMPT = """You are a skill-improvement advisor. Based on the review evidence below, give at most
{max_items} **concrete, actionable** improvement suggestions, ordered by priority; one sentence
each, stating what to change and where. Do not invent issues the evidence does not support.

## Evidence
{evidence}

Output JSON only: {{"suggestions": ["...", "..."]}}"""


def _judge_writing(judge: LLMClient, path: Path, document: str, dimensions: Sequence[str]) -> WritingAssessment:
    prompt = WRITING_PROMPT.format(
        path=path,
        document=document,
        dimension_list="\n".join(f"{i}. {d}" for i, d in enumerate(dimensions, 1)),
    )
    try:
        response = judge.complete([{"role": "user", "content": prompt}], [])
        parsed = _extract_json(response.content)
        dims = tuple(
            WritingDimension(
                name=str(d.get("name", "?")),
                score=max(0.0, min(1.0, float(d.get("score", 0.0)))),
                comment=str(d.get("comment", "")),
            )
            for d in parsed["dimensions"]
        )
        if not dims:
            raise ValueError("judge returned empty dimensions")
        return WritingAssessment(dimensions=dims, overall_comment=str(parsed.get("overall_comment", "")))
    except Exception as e:  # noqa: BLE001 - a judging failure must not block the rest of the review
        return WritingAssessment(dimensions=(), error=f"{type(e).__name__}: {e}")


def _build_evidence(evaluation: SkillEvaluation) -> str:
    evidence: dict[str, Any] = {}
    if evaluation.behavior_results:
        evidence["behavior_tests"] = [
            {
                "case": r.case_name,
                "passed": r.passed,
                "error": r.error,
                "failed_expectations": [
                    {"name": e.name, "score": e.score, "message": e.message}
                    for e in r.expectations
                    if not e.passed
                ],
            }
            for r in evaluation.behavior_results
        ]
    if evaluation.writing and evaluation.writing.dimensions:
        evidence["weak_writing_dimensions"] = [
            {"dimension": d.name, "score": d.score, "comment": d.comment}
            for d in evaluation.writing.dimensions
            if d.score < 0.8
        ]
    return json.dumps(evidence, ensure_ascii=False, indent=2) or "{}"


def _code_suggestions(lint_report: Any, max_items: int) -> list[str]:
    """Deterministic suggestions from lint messages that carry a fix hint,
    errors first (stable within severity: check order is preserved)."""
    if lint_report is None:
        return []
    ordered = [(c.check_id, m) for c in lint_report.results for m in c.messages if m.fix]
    ordered.sort(key=lambda cm: 0 if cm[1].severity is Severity.ERROR else 1)
    return [f"[{check_id}] {m.text} → {m.fix}" for check_id, m in ordered[:max_items]]


def _judge_suggestions(judge: LLMClient, evaluation: SkillEvaluation, max_items: int) -> tuple[list[str], str | None]:
    prompt = SUGGESTIONS_PROMPT.format(max_items=max_items, evidence=_build_evidence(evaluation))
    try:
        response = judge.complete([{"role": "user", "content": prompt}], [])
        parsed = _extract_json(response.content)
        suggestions = [str(s) for s in parsed["suggestions"]][:max_items]
        return suggestions, None
    except Exception as e:  # noqa: BLE001
        return [], f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SkillEvaluationBuilder:
    """Fluent builder returned by evaluate_skill()."""

    skill_path: Path
    judge: LLMClient | None = None
    agent_llm: LLMClient | None = None
    tools: tuple[Tool, ...] = ()
    cases: tuple[SmeltCase, ...] = ()
    lint_enabled: bool = True
    writing_enabled: bool = True
    writing_dimensions: tuple[str, ...] = DEFAULT_DIMENSIONS
    suggestions_enabled: bool = True
    suggestions_max: int = 5
    weight_map: dict[str, float] = field(default_factory=lambda: {"behavior": 0.5, "writing": 0.3, "lint": 0.2})
    times: int | None = None  # explicit repeat count; None = auto (3 for stochastic agents, 1 for deterministic)

    def with_cases(self, *cases: SmeltCase) -> SkillEvaluationBuilder:
        """Attach behavior cases. Cases without an agent auto-bind the skill under
        review (using agent_llm or judge)."""
        return replace(self, cases=self.cases + tuple(cases))

    def with_lint(self, enabled: bool = True) -> SkillEvaluationBuilder:
        return replace(self, lint_enabled=enabled)

    def with_writing(self, *, dimensions: Sequence[str] | None = None, enabled: bool = True) -> SkillEvaluationBuilder:
        kwargs: dict[str, Any] = {"writing_enabled": enabled}
        if dimensions is not None:
            if not dimensions:
                raise ValueError("dimensions must not be empty")
            kwargs["writing_dimensions"] = tuple(dimensions)
        return replace(self, **kwargs)

    def with_suggestions(self, *, max_items: int = 5, enabled: bool = True) -> SkillEvaluationBuilder:
        return replace(self, suggestions_enabled=enabled, suggestions_max=max_items)

    def with_weights(self, *, behavior: float, writing: float, lint: float) -> SkillEvaluationBuilder:
        for name, value in (("behavior", behavior), ("writing", writing), ("lint", lint)):
            if value < 0:
                raise ValueError(f"weight {name} must not be negative")
        return replace(self, weight_map={"behavior": behavior, "writing": writing, "lint": lint})

    def with_times(self, n: int) -> SkillEvaluationBuilder:
        """Run every behavior case ``n`` times and aggregate mean ± std.

        Overrides per-case .repeat() counts and the auto default (3 for
        stochastic agents, 1 for deterministic ones).
        """
        if n < 1:
            raise ValueError(f"times must be >= 1, got {n}")
        return replace(self, times=n)

    def run(self) -> SkillEvaluation:
        doc = load_skill(self.skill_path)
        evaluation = SkillEvaluation(
            skill_path=str(self.skill_path),
            skill_name=doc.name,
            weights=dict(self.weight_map),
        )

        if self.cases:
            evaluation.behavior_results = [
                self._bind(case).run(times=self._times_for(case)) for case in self.cases
            ]
        evaluation.reference_coverage = self._reference_coverage(evaluation.behavior_results)

        if self.lint_enabled:
            evaluation.lint_report = build_report(doc, run_checks(doc))

        if self.writing_enabled:
            if self.judge is None:
                evaluation.writing = WritingAssessment(dimensions=(), error="no judge LLM provided; writing review skipped")
            else:
                evaluation.writing = _judge_writing(
                    self.judge, self.skill_path, self._read_document(), self.writing_dimensions
                )

        if self.suggestions_enabled:
            evaluation.suggestions_ran = True
            code_suggestions = _code_suggestions(evaluation.lint_report, self.suggestions_max)
            if self.judge is None:
                evaluation.suggestions = code_suggestions
                if not self.lint_enabled:
                    evaluation.suggestions_error = "no judge LLM provided; suggestion generation skipped"
            else:
                suggestions = list(code_suggestions)
                if len(suggestions) < self.suggestions_max and _build_evidence(evaluation) != "{}":
                    llm_suggestions, error = _judge_suggestions(self.judge, evaluation, self.suggestions_max)
                    seen = {s.strip() for s in suggestions}
                    for s in llm_suggestions:
                        if s.strip() not in seen:
                            suggestions.append(s)
                            seen.add(s.strip())
                    evaluation.suggestions_error = error
                evaluation.suggestions = suggestions[: self.suggestions_max]

        return evaluation

    def _read_document(self) -> str:
        """Read the raw SKILL.md (frontmatter included — the writing review needs metadata)."""
        path = self.skill_path
        if path.is_dir():
            path = path / "SKILL.md"
        return path.read_text(encoding="utf-8")

    def _bind(self, case: SmeltCase) -> SmeltCase:
        """Cases without an agent get bound to a SmeltAgent loading the skill under review."""
        if case.agent is not None:
            return case
        llm = self.agent_llm or self.judge
        if llm is None:
            raise ValueError(f"case {case.name!r} has no agent, and no agent_llm / judge was provided for auto-binding")
        return case.given(smelt_agent(self.skill_path, llm=llm, tools=self.tools))

    def _times_for(self, case: SmeltCase) -> int:
        """Effective repeat count: explicit builder setting > per-case .repeat() > auto."""
        if self.times is not None:
            return self.times
        if case.times != 1:
            return case.times
        fallback_llm = self.agent_llm or self.judge
        return 1 if _is_deterministic(case, fallback_llm) else DEFAULT_TIMES

    def _reference_coverage(self, results: list[CaseResult]) -> list[dict[str, Any]] | None:
        """Universe = registered refs ∪ SKILL.md scan; reached = tool-call hits
        across behavior traces. None when the universe is empty."""
        from smelt.refpath import calls_reading

        declared = sorted({r for res in results for r in res.references})
        scanned = _scan_skill_refs(self._read_document())
        universe = sorted(set(declared) | set(scanned))
        if not universe:
            return None
        return [
            {
                "path": p,
                "reached": sum(
                    len(calls_reading(res.trace, p))
                    for res in results
                    if res.trace is not None
                ),
                "origin": ("both" if p in declared and p in scanned
                           else "declared" if p in declared else "scanned"),
            }
            for p in universe
        ]


def _is_deterministic(case: SmeltCase, fallback_llm: LLMClient | None) -> bool:
    """A case is deterministic when its agent replays fixed output: FixedAgent,
    or a SmeltAgent driven by ScriptedLLM (whole-agent or llm fragment).
    Custom agents are assumed stochastic."""
    agent = case.agent
    if isinstance(agent, FixedAgent):
        return True
    if isinstance(agent, SmeltAgent):
        return isinstance(agent.llm, ScriptedLLM)
    if agent is None:
        client = next((f.client for f in case.fragments if isinstance(f, LLMSpec)), None) or fallback_llm
        return client is not None and isinstance(client, ScriptedLLM)
    return False


def evaluate_skill(
    skill: str | Path,
    *,
    judge: LLMClient | None = None,
    agent_llm: LLMClient | None = None,
    tools: Sequence[Tool] = (),
) -> SkillEvaluationBuilder:
    """Start a comprehensive skill evaluation.

    - ``judge``: judging model for the writing review and suggestion generation;
    - ``agent_llm``: model used when auto-binding agents for behavior cases
      (falls back to judge);
    - ``tools``: tools available to auto-bound agents.
    """
    return SkillEvaluationBuilder(
        skill_path=Path(skill),
        judge=judge,
        agent_llm=agent_llm,
        tools=tuple(tools),
    )
