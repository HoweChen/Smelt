"""Tests for smelt.lint.markdown tokenization helpers."""

from smelt.lint.markdown import fences, headings, inline_code, link_targets


def test_headings_ignores_hashes_inside_code_fences():
    body = "# Title\n\n```sh\n# not a heading\n## also not\n```\n\n## Real\n"
    assert headings(body) == [(1, "Title"), (2, "Real")]


def test_headings_handles_setext_and_closing_hashes():
    body = "Setext Title\n============\n\n## Closed ##\n"
    assert headings(body) == [(1, "Setext Title"), (2, "Closed")]


def test_fences_returns_info_and_content():
    body = "```python\nprint(1)\n```\n\n```\nplain\n```\n"
    assert fences(body) == [("python", "print(1)\n"), ("", "plain\n")]


def test_inline_code_collects_code_spans():
    body = "Run `scripts/a.py` then `scripts\\b.py`, not [a link](x.md)."
    assert inline_code(body) == ["scripts/a.py", "scripts\\b.py"]


def test_link_targets_can_exclude_images():
    body = "![img](assets/pic.png) and [doc](references/x.md)"
    assert link_targets(body) == ["assets/pic.png", "references/x.md"]
    assert link_targets(body, include_images=False) == ["references/x.md"]
