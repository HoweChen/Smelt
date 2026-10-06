"""Suite: run a whole battery of cases and produce a summary report.

Generates ``index.html`` (pass rate, score distribution, per-case links) plus one
HTML page per case under ``.smelt/reports/suites/<slug>-<timestamp>/``, with a
``<slug>-latest`` copy tracking the newest run.

Usage::

    from smelt import suite

    result = (
        suite("commit-skill v1.2")
        .add(case_a, case_b)
        .report()
    )
    result.assert_passed()
"""

from __future__ import annotations

import html
import re
import shutil
import textwrap
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from smelt.case import SmeltCase
from smelt.report.html import BASE_CSS, render_html
from smelt.results import CaseResult


def _slug(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", name).strip("-") or "suite"


@dataclass
class SuiteResult:
    """Aggregated result of a suite run."""

    suite_name: str
    results: list[CaseResult]
    report_path: str | None = None  # index.html, set when run via .report()

    @property
    def passed(self) -> bool:
        return bool(self.results) and all(r.passed for r in self.results)

    @property
    def score(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.score for r in self.results) / len(self.results)

    @property
    def pass_count(self) -> int:
        return sum(1 for r in self.results if r.passed)

    def summary(self) -> str:
        mark = "✔" if self.passed else "✘"
        lines = [f"{mark} suite {self.suite_name!r}  {self.pass_count}/{len(self.results)} passed  score={self.score:.2f}"]
        lines.extend(f"  {r.summary()}" for r in self.results)
        return "\n".join(lines)

    def assert_passed(self) -> SuiteResult:
        if not self.passed:
            raise AssertionError(self.summary())
        return self


@dataclass(frozen=True)
class Suite:
    """A named battery of cases. Immutable like SmeltCase."""

    name: str
    cases: tuple[SmeltCase, ...] = ()

    def add(self, *cases: SmeltCase) -> Suite:
        for c in cases:
            if not isinstance(c, SmeltCase):
                raise TypeError(f"suite.add() accepts SmeltCase only, got: {type(c).__name__}")
        return replace(self, cases=self.cases + tuple(cases))

    def run(self) -> SuiteResult:
        """Run every case and return the aggregated result."""
        return SuiteResult(suite_name=self.name, results=[c.run() for c in self.cases])

    def report(self, *, output_dir: str | None = None, quiet: bool = False) -> SuiteResult:
        """Run all cases, write index.html + per-case pages, return the result."""
        result = self.run()
        index = write_suite_report(result, output_dir or ".smelt/reports")
        result.report_path = str(index)
        if not quiet:
            print(f"suite report written to {index}")
        return result

    def report_cli(self, *, quiet: bool = False) -> SuiteResult:
        """Run all cases and print a compact terminal summary; return the result."""
        result = self.run()
        if not quiet:
            print(render_suite_text(result))
        return result


def suite(name: str) -> Suite:
    """suite("commit-skill v1.2").add(case_a, case_b).report()"""
    return Suite(name=name)


# ---------------------------------------------------------------------------
# HTML summary page
# ---------------------------------------------------------------------------


def _esc(value: object) -> str:
    return html.escape(str(value))


def render_suite_html(result: SuiteResult, case_links: dict[str, str] | None = None) -> str:
    """Render the suite overview page.

    ``case_links`` maps case name → relative HTML page path; cases without a
    link render as plain text.
    """
    status = "PASS" if result.passed else "FAIL"
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    links = case_links or {}

    buckets = {"1.00": 0, "0.80–0.99": 0, "0.50–0.79": 0, "<0.50": 0}
    for r in result.results:
        if r.score >= 1.0:
            buckets["1.00"] += 1
        elif r.score >= 0.8:
            buckets["0.80–0.99"] += 1
        elif r.score >= 0.5:
            buckets["0.50–0.79"] += 1
        else:
            buckets["<0.50"] += 1

    rows = []
    for r in result.results:
        cls = "pass" if r.passed else "fail"
        mark = "✔" if r.passed else "✘"
        label = _esc(r.case_name)
        if r.case_name in links:
            label = f'<a href="{_esc(links[r.case_name])}">{label}</a>'
        failed = sum(1 for e in r.expectations if not e.passed)
        note = r.error or (f"{failed} expectation(s) failed" if failed else "")
        rows.append(f"""
      <tr>
        <td>{label}</td>
        <td><div class="bar"><div class="bar-fill {cls}" style="width:{round(r.score * 100, 1)}%"></div></div>
            <span class="num">{r.score:.2f}</span></td>
        <td class="{cls}">{mark}</td>
        <td class="msg">{_esc(note)}</td>
      </tr>""")

    dist_cells = "".join(f"<td class='num'>{v}</td>" for v in buckets.values())
    dist_labels = "".join(f"<th>{k}</th>" for k in buckets)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Smelt suite — {_esc(result.suite_name)}</title>
<style>{BASE_CSS}</style>
</head>
<body>
<div class="container">
  <header>
    <span class="badge {status}">{status}</span>
    <h1>{_esc(result.suite_name)}</h1>
    <span>{result.pass_count}/{len(result.results)} passed · score {result.score:.2f}</span>
  </header>
  <div class="meta">generated {generated}</div>
  <section>
    <h2>Score distribution</h2>
    <table>
      <thead><tr>{dist_labels}</tr></thead>
      <tbody><tr>{dist_cells}</tr></tbody>
    </table>
  </section>
  <section>
    <h2>Cases ({len(result.results)})</h2>
    <table>
      <thead><tr><th>Case</th><th>Score</th><th></th><th>Notes</th></tr></thead>
      <tbody>{''.join(rows)}
      </tbody>
    </table>
  </section>
  <footer>generated by smelt</footer>
</div>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Terminal summary
# ---------------------------------------------------------------------------

_WIDTH = 72


def render_suite_text(result: SuiteResult) -> str:
    """Render the suite as a compact terminal summary."""
    status = "PASS" if result.passed else "FAIL"
    lines = [
        "═" * _WIDTH,
        f"  SMELT SUITE  ·  {result.suite_name}",
        "═" * _WIDTH,
        f"  status {status}   {result.pass_count}/{len(result.results)} passed   score {result.score:.2f}",
        "",
    ]
    for r in result.results:
        mark = "✔" if r.passed else "✘"
        note = r.error or ""
        lines.append(f"  {mark} {r.score:.2f}  {r.case_name}")
        if note:
            lines += textwrap.indent(textwrap.fill(note, _WIDTH - 8), "       ").splitlines()
    lines.append("═" * _WIDTH)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def write_suite_report(result: SuiteResult, output_dir: str | Path = ".smelt/reports") -> Path:
    """Write index.html plus one page per case; refresh the ``<slug>-latest`` copy.

    Layout: ``<output_dir>/suites/<slug>-<timestamp>/index.html`` and
    ``<case-slug>.html`` siblings. Returns the index.html path.
    """
    slug = _slug(result.suite_name)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    base = Path(output_dir) / "suites"
    target = base / f"{slug}-{stamp}"
    target.mkdir(parents=True, exist_ok=True)

    links: dict[str, str] = {}
    for r in result.results:
        page = f"{_slug(r.case_name)}.html"
        # disambiguate duplicate case names
        if page in links.values():
            page = f"{_slug(r.case_name)}-{len(links)}.html"
        (target / page).write_text(render_html(r), encoding="utf-8")
        links[r.case_name] = page

    index = target / "index.html"
    index.write_text(render_suite_html(result, links), encoding="utf-8")

    latest = base / f"{slug}-latest"
    if latest.exists():
        shutil.rmtree(latest)
    shutil.copytree(target, latest)
    return index
