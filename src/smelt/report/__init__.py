"""smelt.report — per-case report rendering: HTML files under .smelt/ and
formatted terminal text."""

from smelt.report.html import render_html
from smelt.report.text import render_text
from smelt.report.writer import write_html_report

__all__ = [
    "render_html",
    "render_text",
    "write_html_report",
]
