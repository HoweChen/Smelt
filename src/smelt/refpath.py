"""Workspace-relative reference path matching — shared by then-assertions,
mock_tool, and the evaluate coverage scan."""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import unquote, urlparse

from smelt.trace import ToolCallRecord, Trace


def normalize_ref_path(p: str) -> str:
    """normpath + posix separators; strips leading './'. Backslashes are treated
    as separators too; absolute paths keep their tail so escape spellings still
    trailing-match. Markdown link decorations are removed as well — '#fragment'
    anchors / '?query' suffixes and URL-encoded characters (Chinese anchors) —
    so the SKILL.md scan and the trace matching agree on one spelling."""
    raw = unquote(urlparse(p.strip().replace("\\", "/")).path)
    norm = os.path.normpath(raw).replace(os.sep, "/")
    while norm.startswith("./"):
        norm = norm[2:]
    return norm


def ref_path_matches(value: str, target: str) -> bool:
    t = normalize_ref_path(target)
    v = normalize_ref_path(value)
    return v == t or v.endswith("/" + t)


def _iter_strings(value: Any):
    """Yield every string in a nested argument value (lists/dicts included)."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _iter_strings(v)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_strings(item)


def calls_reading(trace: Trace, path: str) -> list[ToolCallRecord]:
    """Tool calls carrying any string argument (nested included) matching the path."""
    hits = []
    for c in trace.tool_calls:
        if any(ref_path_matches(v, path) for v in _iter_strings(c.arguments)):
            hits.append(c)
    return hits
