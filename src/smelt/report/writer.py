"""Persist HTML reports under the project's .smelt/ directory."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

from smelt.report.html import render_html
from smelt.results import CaseResult


def _slug(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", name).strip("-") or "case"


def write_html_report(result: CaseResult, output_dir: str | Path = ".smelt/reports") -> Path:
    """Write the HTML report for one case.

    Produces ``<slug>-<timestamp>.html`` and refreshes the ``<slug>-latest.html``
    pointer copy. Returns the timestamped path.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = _slug(result.case_name)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    target = out_dir / f"{slug}-{stamp}.html"
    target.write_text(render_html(result), encoding="utf-8")
    latest = out_dir / f"{slug}-latest.html"
    latest.write_text(render_html(result), encoding="utf-8")
    return target
