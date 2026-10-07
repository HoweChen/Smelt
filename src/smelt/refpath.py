"""Workspace-relative reference path matching — shared by then-assertions,
mock_tool, and the evaluate coverage scan."""

from __future__ import annotations

import os

from smelt.trace import ToolCallRecord, Trace


def normalize_ref_path(p: str) -> str:
    """normpath + posix separators; strips leading './'. Absolute paths keep
    their tail so escape spellings still trailing-match."""
    norm = os.path.normpath(p.strip()).replace(os.sep, "/")
    while norm.startswith("./"):
        norm = norm[2:]
    return norm


def ref_path_matches(value: str, target: str) -> bool:
    t = normalize_ref_path(target)
    v = normalize_ref_path(value)
    return v == t or v.endswith("/" + t)


def calls_reading(trace: Trace, path: str) -> list[ToolCallRecord]:
    """Tool calls carrying any string argument matching the reference path."""
    hits = []
    for c in trace.tool_calls:
        if any(isinstance(v, str) and ref_path_matches(v, path) for v in c.arguments.values()):
            hits.append(c)
    return hits
