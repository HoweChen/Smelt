"""Shared CommonMark tokenization helpers for lint checks.

Regex-only scanning of raw markdown cannot tell a link target apart from
prose, which caused false positives (e.g. ``[text](references/x.md)`` being
re-matched as the prose path ``references/x.md)``). Tokenizing once with
markdown-it-py gives checks clean inputs instead.
"""

from __future__ import annotations

from markdown_it import MarkdownIt

_md = MarkdownIt("commonmark")


def link_targets(body: str) -> list[str]:
    """Resolved targets of all links and images, including reference-style links."""
    targets: list[str] = []
    for token in _md.parse(body):
        if token.type != "inline":
            continue
        for child in token.children or []:
            if child.type == "link_open":
                href = child.attrGet("href")
                if href:
                    targets.append(href)
            elif child.type == "image":
                src = child.attrGet("src")
                if src:
                    targets.append(src)
    return targets


def prose_segments(body: str) -> list[str]:
    """Plain-text segments: inline text with code spans and link targets removed.

    Link *text* is kept; only the destination is excluded. Fenced/indented
    code blocks never appear here.
    """
    segments: list[str] = []
    for token in _md.parse(body):
        if token.type != "inline":
            continue
        for child in token.children or []:
            if child.type == "text":
                segments.append(child.content)
    return segments


def code_contents(body: str) -> list[str]:
    """Contents of fenced and indented code blocks."""
    return [t.content for t in _md.parse(body) if t.type in ("fence", "code_block")]
