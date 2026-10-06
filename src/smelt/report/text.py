"""Terminal text report rendering for a single CaseResult."""

from __future__ import annotations

import json
import textwrap

from smelt.results import CaseResult

_WIDTH = 72


def _bar(score: float, width: int = 20) -> str:
    filled = round(score * width)
    return "█" * filled + "░" * (width - filled)


def render_text(result: CaseResult) -> str:
    """Render one CaseResult as a formatted terminal report."""
    status = "PASS" if result.passed else "FAIL"
    spread = f" ±{result.score_std:.2f} (n={result.runs})" if result.runs > 1 else ""
    reliability = ""
    if result.pass_hat is not None:
        reliability = f"  pass^{result.runs} {'✔' if result.pass_hat else '✘'}"
    lines = [
        "═" * _WIDTH,
        f"  SMELT REPORT  ·  {result.case_name}",
        "═" * _WIDTH,
        f"  status    {status}",
        f"  score     {_bar(result.score)} {result.score:.2f}{spread}{reliability}",
        f"  workspace {result.workspace or '-'}",
    ]

    if result.error:
        lines += ["", "  error:"]
        lines += textwrap.indent(textwrap.fill(result.error, _WIDTH - 6), "    ").splitlines()
    for err in result.run_errors:
        lines += ["", "  run error:"]
        lines += textwrap.indent(textwrap.fill(err, _WIDTH - 6), "    ").splitlines()

    lines += ["", "  expectations", "  " + "-" * (_WIDTH - 4)]
    if not result.expectations:
        lines.append("    (none — flow-only run)")
    for e in result.expectations:
        mark = "✔" if e.passed else "✘"
        e_spread = f" ±{e.score_std:.2f} n={e.runs}" if e.runs > 1 else ""
        lines.append(f"    {mark} {_bar(e.score, 10)} {e.score:.2f}≥{e.threshold:.2f}{e_spread}  {e.name}")
        if e.message:
            lines += textwrap.indent(
                textwrap.fill(e.message, _WIDTH - 10), "        "
            ).splitlines()

    if result.trace is not None:
        lines += ["", "  trace", "  " + "-" * (_WIDTH - 4)]
        if result.trace.tool_calls:
            for c in result.trace.tool_calls:
                args = json.dumps(c.arguments, ensure_ascii=False)
                if len(args) > 40:
                    args = args[:37] + "..."
                state = f"error: {c.error}" if c.error else "ok"
                lines.append(f"    → {c.name}({args})  [{state}]")
        else:
            lines.append("    (no tool calls)")
        output = result.trace.output
        lines.append("    final output:")
        lines += textwrap.indent(textwrap.fill(output or "(empty)", _WIDTH - 10), "      ").splitlines()

    lines.append("═" * _WIDTH)
    return "\n".join(lines)
