"""Reference-testing assertions: args negation, reach, restraint, untouched, mock."""

from smelt import fixed_agent, new_case, no_tool_call, text


def test_no_tool_call_with_args_subset():
    agent = fixed_agent("done", tool_calls=[
        {"name": "read_file", "arguments": {"path": "references/endpoints.md"}},
    ])
    blocked = (
        new_case("restraint")
        .given(agent)
        .when(text("go"))
        .then(no_tool_call("read_file", args={"path": "references/endpoints.md"}))
        .run()
    )
    assert not blocked.passed
    assert "references/endpoints.md" in blocked.expectations[0].message

    other_file_ok = (
        new_case("restraint-ok")
        .given(agent)
        .when(text("go"))
        .then(no_tool_call("read_file", args={"path": "references/other.md"}))
        .run()
    )
    assert other_file_ok.passed


def test_no_tool_call_without_args_unchanged():
    agent = fixed_agent("done", tool_calls=[{"name": "read_file", "arguments": {"path": "x"}}])
    result = new_case("n").given(agent).when(text("go")).then(no_tool_call("read_file")).run()
    assert not result.passed  # name-only still blocks any call of that tool


# ---------------------------------------------------------------------------
# Task 2: reference() / reference_folder() fragments + registration
# ---------------------------------------------------------------------------

from pathlib import Path

import pytest

from smelt import output_equals
from smelt.given.references import reference, reference_folder
from smelt.trace import Trace


def _skill(tmp_path) -> Path:
    d = tmp_path / "skills" / "demo"
    (d / "references").mkdir(parents=True)
    (d / "scripts").mkdir()
    (d / "SKILL.md").write_text("---\nname: demo\ndescription: x\n---\nbody", encoding="utf-8")
    (d / "references" / "a.md").write_text("alpha content line one", encoding="utf-8")
    (d / "references" / "b.md").write_text("beta content line two", encoding="utf-8")
    (d / "scripts" / "s.py").write_text("print('hi')", encoding="utf-8")
    return d


def _spy(captured):
    class Spy:
        def run(self, ctx, input):
            captured["refs"] = sorted(ctx.references)
            captured["ws"] = ctx.workspace
            return Trace(output="done")

    return Spy()


def test_reference_folder_mounts_and_registers(tmp_path):
    skill = _skill(tmp_path)
    captured = {}
    result = (
        new_case("mount")
        .given(reference_folder(skill))
        .given(_spy(captured))
        .when(text("go"))
        .then(output_equals("done"))
        .run()
    )
    assert result.passed
    ws = captured["ws"]
    assert (ws / "references" / "a.md").read_text(encoding="utf-8") == "alpha content line one"
    assert (ws / "scripts" / "s.py").exists()
    assert captured["refs"] == ["references/a.md", "references/b.md", "scripts/s.py"]


def test_reference_folder_bare_directory(tmp_path):
    refs = tmp_path / "refs"
    refs.mkdir()
    (refs / "x.md").write_text("x", encoding="utf-8")
    captured = {}
    new_case("bare").given(reference_folder(refs)).given(_spy(captured)).when(text("go")).run()
    assert captured["refs"] == ["refs/x.md"]
    assert (captured["ws"] / "refs" / "x.md").exists()


def test_reference_single_file_skill_relative(tmp_path):
    skill = _skill(tmp_path)
    captured = {}
    new_case("single").given(reference(skill / "references" / "a.md")).given(_spy(captured)).when(text("go")).run()
    assert captured["refs"] == ["references/a.md"]
    assert (captured["ws"] / "references" / "a.md").exists()
    assert not (captured["ws"] / "references" / "b.md").exists()


def test_reference_missing_path_raises_eagerly(tmp_path):
    with pytest.raises(FileNotFoundError, match="ghost"):
        reference(tmp_path / "ghost.md")


def test_reference_folder_missing_dir_raises_eagerly(tmp_path):
    with pytest.raises(FileNotFoundError, match="ghost"):
        reference_folder(tmp_path / "ghost")


def test_stacked_references_accumulate(tmp_path):
    skill = _skill(tmp_path)
    captured = {}
    result = (
        new_case("stacked")
        .given(reference(skill / "references" / "a.md"))
        .given(reference(skill / "references" / "b.md"))
        .given(_spy(captured))
        .when(text("go"))
        .then(output_equals("done"))
        .run()
    )
    assert result.passed
    assert captured["refs"] == ["references/a.md", "references/b.md"]  # merge accumulates


def test_reference_folder_empty_resources_registers_nothing(tmp_path):
    d = tmp_path / "skills" / "empty"
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text("---\nname: e\ndescription: x\n---\nbody", encoding="utf-8")
    captured = {}
    new_case("empty").given(reference_folder(d)).given(_spy(captured)).when(text("go")).run()
    assert captured["refs"] == []


# ---------------------------------------------------------------------------
# Task 3: reference_read / no_reference_read
# ---------------------------------------------------------------------------

from smelt import no_reference_read, reference_read


def test_reference_read_matches_any_tool_and_spelling():
    agent = fixed_agent("done", tool_calls=[
        {"name": "run_command", "arguments": {"cmd": "cat ./references/endpoints.md"}},
    ])
    result = (
        new_case("reach")
        .given(agent)
        .when(text("go"))
        .then(reference_read("references/endpoints.md"))
        .run()
    )
    assert result.passed  # matched via a shell command arg with ./ spelling


def test_reference_read_detects_absolute_escape():
    agent = fixed_agent("done", tool_calls=[
        {"name": "read_file", "arguments": {"path": "/real/disk/skills/x/references/endpoints.md"}},
    ])
    result = new_case("escape").given(agent).when(text("go")).then(reference_read("references/endpoints.md")).run()
    assert result.passed  # escape attempt is detected, not missed


def test_reference_read_fails_when_unread():
    result = (
        new_case("miss")
        .given(fixed_agent("done"))
        .when(text("go"))
        .then(reference_read("references/endpoints.md"))
        .run()
    )
    assert not result.passed


def test_no_reference_read():
    ok = new_case("r1").given(fixed_agent("done")).when(text("go")).then(no_reference_read("references/x.md")).run()
    assert ok.passed
    bad_agent = fixed_agent("done", tool_calls=[{"name": "read_file", "arguments": {"path": "references/x.md"}}])
    bad = new_case("r2").given(bad_agent).when(text("go")).then(no_reference_read("references/x.md")).run()
    assert not bad.passed


# ---------------------------------------------------------------------------
# Task 4: reference_untouched (content-fingerprint ablation validity)
# ---------------------------------------------------------------------------

from smelt import reference_untouched

CONTENT = "The limit parameter accepts values from 1 to 100 inclusive.\nShort.\nThe after parameter is a fullname cursor for pagination."


def _skill_with_ref(tmp_path):
    d = tmp_path / "skills" / "demo"
    (d / "references").mkdir(parents=True)
    (d / "references" / "endpoints.md").write_text(CONTENT, encoding="utf-8")
    return d


def test_reference_untouched_passes_when_never_consulted(tmp_path):
    skill = _skill_with_ref(tmp_path)
    result = (
        new_case("clean")
        .given(fixed_agent("I do not know the parameters."))
        .when(text("go"))
        .then(reference_untouched("references/endpoints.md", source=skill))
        .run()
    )
    assert result.passed


def test_reference_untouched_fails_on_direct_read(tmp_path):
    skill = _skill_with_ref(tmp_path)
    agent = fixed_agent("done", tool_calls=[{"name": "read_file", "arguments": {"path": "references/endpoints.md"}}])
    result = new_case("direct").given(agent).when(text("go")).then(
        reference_untouched("references/endpoints.md", source=skill)).run()
    assert not result.passed


def test_reference_untouched_fails_on_detour_content_leak(tmp_path):
    # never "read" the path, but a shell command's RESULT contains the content
    skill = _skill_with_ref(tmp_path)
    agent = fixed_agent("done", tool_calls=[
        {"name": "run_command", "arguments": {"cmd": "grep -r limit ."}, "result": CONTENT},
    ])
    result = new_case("detour").given(agent).when(text("go")).then(
        reference_untouched("references/endpoints.md", source=skill)).run()
    assert not result.passed
    assert "leaked" in result.expectations[0].message


def test_reference_untouched_flags_parametric_memory(tmp_path):
    skill = _skill_with_ref(tmp_path)
    agent = fixed_agent("The limit parameter accepts values from 1 to 100 inclusive.")  # no tool calls at all
    result = new_case("memory").given(agent).when(text("go")).then(
        reference_untouched("references/endpoints.md", source=skill)).run()
    assert not result.passed
    assert "contamination" in result.expectations[0].message


def test_reference_untouched_missing_source_scores_zero(tmp_path):
    result = (
        new_case("nosource")
        .given(fixed_agent("x"))
        .when(text("go"))
        .then(reference_untouched("references/ghost.md", source=tmp_path / "nope"))
        .run()
    )
    assert not result.passed
    assert "source not found" in result.expectations[0].message


# ---------------------------------------------------------------------------
# Task 5: mock_tool (content-level ablation / robustness)
# ---------------------------------------------------------------------------

from smelt import mock_tool, tool


def test_mock_tool_intercepts_matching_path(tmp_path):
    real = tmp_path / "real.md"
    real.write_text("real content", encoding="utf-8")

    @tool
    def read_file(path: str) -> str:
        """Read a file"""
        return Path(path).read_text(encoding="utf-8")

    mocked = mock_tool(read_file, {"references/endpoints.md": ""})
    assert mocked.name == "read_file"
    assert mocked.spec()["parameters"] == read_file.spec()["parameters"]
    assert mocked.invoke({"path": "references/endpoints.md"}) == ""
    assert mocked.invoke({"path": str(real)}) == "real content"  # non-matching delegates
