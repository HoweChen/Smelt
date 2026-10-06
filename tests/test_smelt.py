"""Core framework tests: DSL, agents, expectations, runner."""

from pathlib import Path

import pytest

from smelt import (
    LLMResponse,
    ScriptedLLM,
    context,
    directory,
    fixed_agent,
    json_output,
    new_case,
    no_tool_call,
    output_contains,
    output_equals,
    smelt_agent,
    text,
    text_similar,
    tool,
    tool_call,
)
from smelt.pytest_plugin import SmeltSession

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "examples" / "fixtures"


# ---------------------------------------------------------------------------
# DSL structure
# ---------------------------------------------------------------------------


def test_dsl_is_immutable_and_chainable():
    base = new_case("c1").given(fixed_agent("done"))
    derived = base.when(text("hi")).then(output_equals("done"))
    assert base.trigger is None  # the original instance is untouched
    assert derived.trigger is not None
    assert derived.run().passed


def test_given_rejects_second_agent():
    case = new_case("c").given(fixed_agent("a"))
    with pytest.raises(ValueError, match="only one agent per case"):
        case.given(fixed_agent("b"))


def test_when_rejects_second_trigger():
    case = new_case("c").given(fixed_agent("a")).when(text("x"))
    with pytest.raises(ValueError, match="only one trigger per case"):
        case.when(text("y"))


def test_missing_agent_and_trigger_converge_to_error_result():
    r1 = new_case("no-agent").when(text("x")).then(output_equals("y")).run()
    assert not r1.passed and "missing agent" in r1.error
    r2 = new_case("no-when").given(fixed_agent("y")).then(output_equals("y")).run()
    assert not r2.passed and "missing trigger" in r2.error


# ---------------------------------------------------------------------------
# Contexts and inputs
# ---------------------------------------------------------------------------


def test_context_materializes_files_into_workspace(tmp_path):
    src = tmp_path / "fixture"
    (src / "sub").mkdir(parents=True)
    (src / "sub" / "a.txt").write_text("hello", encoding="utf-8")

    result = (
        new_case("ctx")
        .given(context(files=src, prompt="background", env={"SMELT_X": "1"}))
        .given(fixed_agent("ok"))
        .when(text("go"))
        .then(output_equals("ok"))
        .run()
    )
    assert result.passed
    copied = Path(result.workspace) / "sub" / "a.txt"
    assert copied.read_text(encoding="utf-8") == "hello"


def test_directory_input_copies_and_lists(tmp_path):
    src = tmp_path / "indir"
    src.mkdir()
    (src / "note.md").write_text("content", encoding="utf-8")

    captured = {}

    class ProbeAgent:
        def run(self, ctx, input):
            captured["rendered"] = input.render(ctx.workspace)
            captured["exists"] = (ctx.workspace / "note.md").exists()
            from smelt import Trace

            return Trace(output="ok")

    result = (
        new_case("dir-input")
        .given(ProbeAgent())
        .when(directory(src))
        .then(output_equals("ok"))
        .run()
    )
    assert result.passed
    assert captured["exists"]
    assert "note.md" in captured["rendered"]


# ---------------------------------------------------------------------------
# Expectations (then)
# ---------------------------------------------------------------------------


def _run_fixed(output="", calls=(), *expectations):
    return (
        new_case("fixed")
        .given(fixed_agent(output, tool_calls=calls))
        .when(text("go"))
        .then(*expectations)
        .run()
    )


def test_tool_call_match_and_partial_score():
    calls = [{"name": "run_command", "arguments": {"cmd": "git status", "cwd": "."}}]
    ok = _run_fixed("", calls, tool_call("run_command", args={"cmd": "git status"}))
    assert ok.passed and ok.expectations[0].score == 1.0

    partial = _run_fixed("", calls, tool_call("run_command", args={"cmd": "git push"}))
    assert not partial.passed and partial.expectations[0].score == 0.5

    none = _run_fixed("", (), tool_call("run_command"))
    assert not none.passed and none.expectations[0].score == 0.0


def test_no_tool_call():
    calls = [{"name": "delete_everything", "arguments": {}}]
    assert _run_fixed("", calls, no_tool_call("format_disk")).passed
    assert not _run_fixed("", calls, no_tool_call("delete_everything")).passed


def test_output_equals_and_contains_partial():
    assert _run_fixed("done", (), output_equals("done")).passed
    r = _run_fixed("committed abc", (), output_contains("committed", "xyz", threshold=0.5))
    assert r.passed and r.expectations[0].score == 0.5


def test_text_similar_threshold():
    r = _run_fixed("I have committed all changes for you", (), text_similar("I committed all changes for you", threshold=0.8))
    assert r.expectations[0].score > 0.8
    assert r.passed

    low = _run_fixed("a completely unrelated answer", (), text_similar("I committed all changes for you", threshold=0.8))
    assert not low.passed


def test_text_similar_custom_scorer():
    scorer = lambda a, b: 1.0 if set(a) & set(b) else 0.0
    r = _run_fixed("abc", (), text_similar("xyz", threshold=0.9, scorer=scorer))
    assert not r.passed  # no overlap
    r2 = _run_fixed("abc", (), text_similar("cde", threshold=0.9, scorer=scorer))
    assert r2.passed  # overlap on 'c'


def test_json_output_schema_and_contains():
    schema = {
        "type": "object",
        "required": ["name", "tags"],
        "properties": {"name": {"type": "string"}, "tags": {"type": "array", "items": {"type": "string"}}},
    }
    good = _run_fixed('{"name": "smelt", "tags": ["a"]}', (), json_output(schema, contains={"name": "smelt"}))
    assert good.passed and good.expectations[0].score == 1.0

    missing_field = _run_fixed('{"name": "smelt"}', (), json_output(schema))
    assert not missing_field.passed

    fenced = _run_fixed('Here you go:\n```json\n{"ok": true}\n```', (), json_output(contains={"ok": True}))
    assert fenced.passed

    garbage = _run_fixed("no json here", (), json_output())
    assert not garbage.passed and garbage.expectations[0].score == 0.0


# ---------------------------------------------------------------------------
# SmeltAgent + ScriptedLLM: the full tool loop
# ---------------------------------------------------------------------------


def _make_tools():
    @tool
    def write_file(path: str, content: str) -> str:
        """Write a file into the workspace"""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"wrote {p}"

    return [write_file]


def test_smelt_agent_tool_loop_runs_in_workspace(tmp_path):
    llm = ScriptedLLM([
        LLMResponse.call("write_file", {"path": "out/result.txt", "content": "42"}),
        LLMResponse.say("result.txt written"),
    ])
    skill = tmp_path / "SKILL.md"
    skill.write_text("---\nname: writer\ndescription: write files\n---\n\n# Writer skill\n", encoding="utf-8")

    result = (
        new_case("agent-loop", keep_workspace=True)
        .given(smelt_agent(skill, llm=llm, tools=_make_tools()))
        .when(text("write 42 into out/result.txt"))
        .then(tool_call("write_file", args={"path": "out/result.txt"}))
        .then(output_contains("written"))
        .run()
    )
    assert result.passed, result.summary()
    assert (Path(result.workspace) / "out" / "result.txt").read_text() == "42"
    # the tool ran inside the workspace, not polluting the project directory
    assert not (ROOT / "out" / "result.txt").exists()


def test_smelt_agent_unknown_tool_goes_into_trace():
    llm = ScriptedLLM([
        LLMResponse.call("nonexistent"),
        LLMResponse.say("oh well"),
    ])
    result = (
        new_case("unknown-tool")
        .given(smelt_agent(llm=llm, tools=[], system_prompt="sys"))
        .when(text("hi"))
        .then(tool_call("nonexistent"))
        .then(output_equals("oh well"))
        .run()
    )
    assert result.passed
    assert result.trace.tool_calls[0].error.startswith("unknown tool")


# ---------------------------------------------------------------------------
# pytest plugin
# ---------------------------------------------------------------------------


def test_smelt_session_check_raises_on_failure():
    session = SmeltSession()
    good = new_case("g").given(fixed_agent("x")).when(text("go")).then(output_equals("x"))
    assert session.check(good).passed

    bad = new_case("b").given(fixed_agent("x")).when(text("go")).then(output_equals("y"))
    with pytest.raises(AssertionError, match="case 'b'"):
        session.check(bad)
    assert len(session.results) == 2


# ---------------------------------------------------------------------------
# CLI run
# ---------------------------------------------------------------------------


def test_cli_run_collects_cases(tmp_path, capsys):
    case_file = tmp_path / "demo_cases.py"
    case_file.write_text(
        "from smelt import new_case, fixed_agent, text, output_equals\n"
        "case_ok = new_case('ok').given(fixed_agent('done')).when(text('go')).then(output_equals('done'))\n"
        "case_bad = new_case('bad').given(fixed_agent('done')).when(text('go')).then(output_equals('nope'))\n",
        encoding="utf-8",
    )
    from smelt.cli import main

    assert main(["run", str(case_file)]) == 1
    out = capsys.readouterr().out
    assert "1/2 cases passed" in out

    case_file.write_text(
        "from smelt import new_case, fixed_agent, text, output_equals\n"
        "case_ok = new_case('ok').given(fixed_agent('done')).when(text('go')).then(output_equals('done'))\n",
        encoding="utf-8",
    )
    assert main(["run", str(case_file)]) == 0
