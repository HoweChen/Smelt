"""Report rendering: terminal text / Markdown / JSON."""

from __future__ import annotations

import json
from dataclasses import asdict

from smelt.lint.models import Severity, SkillReport

_ICON = {Severity.INFO: "ℹ", Severity.WARNING: "⚠", Severity.ERROR: "✖"}


def render_text(report: SkillReport) -> str:
    lines = [
        (
            f"■ {report.skill_path.name}  score {report.total_score:.1f}  grade {report.grade}"
            f"{' (PASS)' if report.passed else ' (FAIL)'}"
        ),
        "",
    ]
    for r in report.results:
        mark = "✔" if r.passed else "✖"
        lines.append(f"  {mark} [{r.score:5.1f}] {r.name}")
        for m in r.messages:
            lines.append(f"        {_ICON[m.severity]} {m.text}")
    lines.append("")
    return "\n".join(lines)


def render_markdown(reports: list[SkillReport]) -> str:
    parts = ["# Skill Quality Report", ""]
    for report in reports:
        status = "✅ PASS" if report.passed else "❌ FAIL"
        parts.append(f"## {report.skill_path.name} — {report.total_score:.1f} pts ({report.grade}) {status}")
        parts.append("")
        parts.append("| Check | Score | Result | Issues |")
        parts.append("|---|---|---|---|")
        for r in report.results:
            msgs = "<br>".join(f"{_ICON[m.severity]} {m.text}" for m in r.messages) or "—"
            parts.append(f"| {r.name} | {r.score:.1f} | {'✅' if r.passed else '❌'} | {msgs} |")
        parts.append("")
    return "\n".join(parts)


def render_json(reports: list[SkillReport]) -> str:
    payload = [
        {
            "skill": str(r.skill_path),
            "total_score": r.total_score,
            "grade": r.grade,
            "passed": r.passed,
            "checks": [
                {
                    **{k: v for k, v in asdict(c).items() if k != "messages"},
                    "messages": [{"severity": m.severity.value, "text": m.text} for m in c.messages],
                }
                for c in r.results
            ],
        }
        for r in reports
    ]
    return json.dumps(payload, ensure_ascii=False, indent=2)
