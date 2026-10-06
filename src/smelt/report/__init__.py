"""smelt.report — per-case and suite report rendering: HTML files under .smelt/
and formatted terminal text."""

from smelt.report.html import render_html
from smelt.report.suite import (
    Suite,
    SuiteResult,
    render_suite_html,
    render_suite_text,
    suite,
    write_suite_report,
)
from smelt.report.text import render_text
from smelt.report.writer import write_html_report

__all__ = [
    "Suite",
    "SuiteResult",
    "render_html",
    "render_suite_html",
    "render_suite_text",
    "render_text",
    "suite",
    "write_html_report",
    "write_suite_report",
]
