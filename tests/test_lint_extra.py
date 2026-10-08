"""Supplementary tests for the lint submodule and assorted coverage gaps."""

import os
import sys
from pathlib import Path

import pytest

from smelt.lint.checks.metadata import MetadataCheck
from smelt.lint.loader import (
    SkillLoadError,
    _parse_simple_yaml,
    discover_skills,
    load_skill,
    parse_frontmatter,
)
from smelt.lint.models import CheckResult, Message, Severity, SkillDoc
from smelt.lint.scorer import build_report, grade_of

ROOT = Path(__file__).resolve().parent.parent
GOOD = ROOT / "examples" / "good-skill"


# ---------------------------------------------------------------------------
# loader: the simple YAML parser and error paths
# ---------------------------------------------------------------------------


def test_parse_simple_yaml_all_branches():
    text = """
    # a comment line
    name: "quoted"
    tags: [a, 'b', c]
    empty_value:
    items:
      - first
      - 'second'
    line without a colon

    count: 42
    """
    data = _parse_simple_yaml(text)
    assert data["name"] == "quoted"
    assert data["tags"] == ["a", "b", "c"]
    assert data["empty_value"] is None
    assert data["items"] == ["first", "second"]
    assert data["count"] == "42"


def test_parse_frontmatter_uses_pyyaml_when_available():
    pytest.importorskip("yaml")
    fm, _ = parse_frontmatter("---\nname: x\nnested:\n  a: 1\n---\nbody")
    assert fm["name"] == "x"


def test_load_skill_missing_file_raises(tmp_path):
    with pytest.raises(SkillLoadError, match="SKILL.md not found"):
        load_skill(tmp_path)


def test_discover_skills_errors(tmp_path):
    with pytest.raises(SkillLoadError, match="does not exist or is not a directory"):
        discover_skills(tmp_path / "ghost")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SkillLoadError, match="no skill directories"):
        discover_skills(empty)
    # pointing directly at a directory containing SKILL.md
    found = discover_skills(GOOD)
    assert found == [GOOD]


def test_load_skill_from_file_path():
    doc = load_skill(GOOD / "SKILL.md")
    assert doc.name == "good-skill"


# ---------------------------------------------------------------------------
# metadata check: every deduction branch
# ---------------------------------------------------------------------------


def _doc(frontmatter, body: str = "body") -> SkillDoc:
    return SkillDoc(
        path=GOOD / "SKILL.md",
        name=frontmatter.get("name", "x") if frontmatter else "x",
        description=frontmatter.get("description", "") if frontmatter else "",
        frontmatter=frontmatter,
        body=body,
    )


def test_metadata_no_frontmatter_scores_zero():
    result = MetadataCheck().run(_doc({}))
    assert result.score == 0.0
    assert any(m.severity == Severity.ERROR and "frontmatter" in m.text for m in result.messages)


def test_metadata_missing_name_and_description():
    result = MetadataCheck().run(_doc({"other": "x"}))
    texts = [m.text for m in result.messages]
    assert any("missing the name field" in t for t in texts)
    assert any("missing the description field" in t for t in texts)
    assert result.score <= 20


def test_metadata_name_mismatch_with_dir():
    doc = _doc({"name": "not-the-dir-name", "description": "a sufficiently long description " * 3})
    result = MetadataCheck().run(doc)
    assert any("does not match the directory name" in m.text for m in result.messages)


def test_metadata_description_too_short_and_too_long():
    short = MetadataCheck().run(_doc({"name": "good-skill", "description": "short"}))
    assert any("too short" in m.text for m in short.messages)

    long = MetadataCheck().run(_doc({"name": "good-skill", "description": "long" * 200}))
    assert any("too long" in m.text for m in long.messages)


# ---------------------------------------------------------------------------
# lint models / scorer edges
# ---------------------------------------------------------------------------


def test_severity_max_empty_and_ordering():
    assert Severity.max([]) == Severity.INFO
    assert Severity.max([Severity.INFO, Severity.ERROR, Severity.WARNING]) == Severity.ERROR


def test_message_str_and_check_result_severity():
    msg = Message(Severity.WARNING, "watch out")
    assert "⚠" in str(msg) and "watch out" in str(msg)
    result = CheckResult(check_id="c", name="n", score=50, passed=False, messages=[msg])
    assert result.severity == Severity.WARNING


def test_scorer_weight_boundaries():
    doc = load_skill(GOOD)
    report = build_report(doc, [])
    assert report.total_score >= 0  # empty results don't crash
    assert grade_of(100) == "A" and grade_of(60) == "D" and grade_of(0) == "F"


# ---------------------------------------------------------------------------
# CLI validate passthrough
# ---------------------------------------------------------------------------


def test_main_validate_passthrough(capsys):
    from smelt.cli import main

    assert main(["validate", str(GOOD)]) == 0
    assert "good-skill" in capsys.readouterr().out


def test_cli_run_unsupported_extension_returns_2(tmp_path, capsys):
    weird = tmp_path / "cases.txt"
    weird.write_text("x = 1", encoding="utf-8")
    from smelt.cli import main

    assert main(["run", str(weird)]) == 2
    assert "failed" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Scattered core-module gaps
# ---------------------------------------------------------------------------


def test_validate_schema_falls_back_without_jsonschema(monkeypatch):
    from smelt import Trace, json_output

    monkeypatch.setitem(sys.modules, "jsonschema", None)
    exp = json_output({"type": "object", "required": ["a"]})
    assert exp.evaluate(Trace(output='{"a": 1}')).passed
    assert not exp.evaluate(Trace(output="{}")).passed


def test_context_directory_mapped_to_subdir(tmp_path):
    from smelt import context, fixed_agent, new_case, output_equals, text

    src = tmp_path / "srcdir"
    src.mkdir()
    (src / "f.txt").write_text("x", encoding="utf-8")
    result = (
        new_case("map-dir")
        .given(context(files={"sub/dir": src}))
        .given(fixed_agent("ok"))
        .when(text("go"))
        .then(output_equals("ok"))
        .run()
    )
    assert (Path(result.workspace) / "sub" / "dir" / "f.txt").exists()


def test_case_result_summary_includes_error():
    from smelt.results import CaseResult

    summary = CaseResult(case_name="e", expectations=[], error="boom").summary()
    assert "error: boom" in summary


def test_openai_client_minimal_kwargs(monkeypatch):
    import types

    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            msg = types.SimpleNamespace(content="ok", tool_calls=None)
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured["init"] = kwargs
            self.chat = types.SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    from smelt.given.agents.llm import OpenAIChatClient

    client = OpenAIChatClient("m")
    resp = client.complete([], [])
    assert captured["init"] == {}  # base_url / api_key omitted when unset
    assert resp.content == "ok" and resp.tool_calls == ()


def test_env_restore_preserves_preexisting_value():
    from smelt import LLMResponse, ScriptedLLM, context, new_case, output_equals, smelt_agent, text, tool

    @tool
    def noop() -> str:
        """No-op"""
        return "ok"

    os.environ["SMELT_PRESET"] = "original"
    try:
        llm = ScriptedLLM([LLMResponse.call("noop"), LLMResponse.say("done")])
        result = (
            new_case("env-restore")
            .given(context(env={"SMELT_PRESET": "temp"}))
            .given(smelt_agent(llm=llm, tools=[noop], system_prompt="s"))
            .when(text("go"))
            .then(output_equals("done"))
            .run()
        )
        assert result.passed
        assert os.environ["SMELT_PRESET"] == "original"  # restored, not deleted
    finally:
        os.environ.pop("SMELT_PRESET", None)
