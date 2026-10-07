"""Tests for the smelt.lint static-lint submodule."""

import json
from pathlib import Path

from smelt.lint.checks import run_checks
from smelt.lint.cli import main
from smelt.lint.loader import count_words, discover_skills, load_skill, parse_frontmatter
from smelt.lint.models import Severity
from smelt.lint.report import render_json, render_markdown, render_text
from smelt.lint.scorer import build_report, grade_of

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"
GOOD = EXAMPLES / "good_skill"
BAD = EXAMPLES / "bad_skill"


def test_count_words_mixed_cjk_and_latin():
    # 1 Latin word + 2 CJK chars
    assert count_words("hello 世界") == 1 + 2


def test_parse_frontmatter():
    fm, body = parse_frontmatter("---\nname: x\ndescription: some description\n---\n\n# Title\nBody")
    assert fm["name"] == "x"
    assert body.lstrip().startswith("# Title")


def test_parse_frontmatter_without_yaml_block():
    fm, body = parse_frontmatter("# no frontmatter")
    assert fm == {}
    assert "no frontmatter" in body


def test_load_good_skill():
    doc = load_skill(GOOD)
    assert doc.name == "good_skill"
    assert doc.description.startswith("Use this skill")
    assert len(doc.sections) >= 4
    assert doc.word_count >= 250


def test_discover_skills_scans_children():
    found = {p.name for p in discover_skills(EXAMPLES)}
    assert found == {"good_skill", "bad_skill", "bad_ref_skill"}


def test_good_skill_scores_grade_a():
    doc = load_skill(GOOD)
    report = build_report(doc, run_checks(doc))
    assert report.grade == "A"
    assert all(r.passed for r in report.results)


def test_bad_skill_fails_and_flags_issues():
    doc = load_skill(BAD)
    report = build_report(doc, run_checks(doc))
    assert report.grade == "F"
    failed = {r.check_id for r in report.results if not r.passed}
    # these three checks fail outright
    assert {"structure", "trigger", "testcoverage"} <= failed
    # clarity / assets / metadata stay above the line but must surface ERROR/WARNING
    by_id = {r.check_id: r for r in report.results}
    assert any(m.severity == Severity.ERROR for m in by_id["clarity"].messages)
    assert any(m.severity == Severity.ERROR for m in by_id["assets"].messages)
    assert any(m.severity == Severity.WARNING for m in by_id["metadata"].messages)


def test_grade_of_boundaries():
    assert grade_of(90) == "A"
    assert grade_of(89.9) == "B"
    assert grade_of(59.9) == "F"


def test_cli_validate_good_only_passes(capsys):
    assert main(["validate", str(GOOD)]) == 0
    out = capsys.readouterr().out
    assert "good_skill" in out and "PASS" in out


def test_cli_validate_examples_fails_due_to_bad_skill(capsys):
    assert main(["validate", str(EXAMPLES)]) == 1
    out = capsys.readouterr().out
    assert "bad_skill" in out and "FAIL" in out


def test_cli_missing_path_returns_2(capsys):
    assert main(["validate", str(EXAMPLES / "nope")]) == 2
    assert "error" in capsys.readouterr().err


def test_renderers_smoke():
    doc = load_skill(GOOD)
    report = build_report(doc, run_checks(doc))
    assert "good_skill" in render_text(report)
    assert "| Check |" in render_markdown([report])
    payload = json.loads(render_json([report]))
    assert payload[0]["grade"] == "A"
