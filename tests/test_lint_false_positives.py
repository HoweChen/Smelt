"""Regression tests for downstream-reported false positives (2026-10-08).

1. assets: PROSE_PATH_RE re-matched markdown link targets with a trailing ")",
   producing phantom "referenced file does not exist" errors.
2. clarity: the {{...}} placeholder regex scanned fenced code blocks, flagging
   Vue template interpolation as unfilled placeholders.
3. packaging: smelt.__version__ drifted from pyproject.toml's version.
"""

import tomllib
from pathlib import Path

import smelt
from smelt.lint.checks.assets import AssetCheck
from smelt.lint.checks.clarity import ClarityCheck
from smelt.lint.models import SkillDoc

ROOT = Path(__file__).resolve().parent.parent


def _skill(tmp_path: Path, body: str, files: tuple[str, ...] = ()) -> SkillDoc:
    for rel in files:
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x")
    return SkillDoc(path=tmp_path / "SKILL.md", name="t", description="", frontmatter={}, body=body)


# ---------------------------------------------------------------------------
# assets: markdown link targets must not be re-matched as prose paths
# ---------------------------------------------------------------------------


def test_assets_markdown_link_to_existing_file_not_flagged(tmp_path):
    doc = _skill(
        tmp_path,
        "See [the guide](references/guide.md) for details.",
        files=("references/guide.md",),
    )
    result = AssetCheck().run(doc)
    assert result.messages == []
    assert result.score == 100.0


def test_assets_reference_style_link_to_existing_file_not_flagged(tmp_path):
    body = "See [the guide][g].\n\n[g]: references/guide.md\n"
    doc = _skill(tmp_path, body, files=("references/guide.md",))
    result = AssetCheck().run(doc)
    assert result.messages == []
    assert result.score == 100.0


def test_assets_markdown_link_to_missing_file_still_flagged(tmp_path):
    doc = _skill(tmp_path, "See [the guide](references/absent.md) for details.")
    result = AssetCheck().run(doc)
    assert any("references/absent.md" in m.text for m in result.messages)
    assert all("absent.md)" not in m.text for m in result.messages)


def test_assets_bare_prose_path_to_missing_file_still_flagged(tmp_path):
    doc = _skill(tmp_path, "See references/absent.md for details.")
    result = AssetCheck().run(doc)
    assert any("references/absent.md" in m.text for m in result.messages)


# ---------------------------------------------------------------------------
# clarity: template-interpolation inside code blocks is not a placeholder
# ---------------------------------------------------------------------------

VUE_BODY = (
    "Use interpolation like this:\n\n"
    "```vue\n"
    "<template>\n"
    "  <p>{{ message }}</p>\n"
    "</template>\n"
    "```\n"
)


def test_clarity_vue_interpolation_in_fenced_block_not_flagged(tmp_path):
    result = ClarityCheck().run(_skill(tmp_path, VUE_BODY))
    assert result.messages == []
    assert result.score == 100.0


def test_clarity_unfilled_placeholder_in_prose_still_flagged(tmp_path):
    result = ClarityCheck().run(_skill(tmp_path, "Fill in {{your_name}} before shipping."))
    assert any("placeholder" in m.text for m in result.messages)
    assert result.score < 100.0


def test_clarity_todo_in_code_block_still_flagged(tmp_path):
    body = "Example:\n\n```sh\n# TODO: replace with real command\n```\n"
    result = ClarityCheck().run(_skill(tmp_path, body))
    assert any("TODO" in m.text for m in result.messages)


# ---------------------------------------------------------------------------
# packaging: __version__ must stay in sync with pyproject.toml
# ---------------------------------------------------------------------------


def test_version_matches_pyproject():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert smelt.__version__ == data["project"]["version"]
