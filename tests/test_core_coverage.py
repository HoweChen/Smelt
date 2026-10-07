"""Supplementary coverage tests for core modules: context / inputs / case / runner /
tools / trace / results / llm / expectations."""

import os
import sys
import types
from pathlib import Path

import pytest

from smelt import (
    LLMResponse,
    ScriptedLLM,
    ToolCall,
    context,
    directory,
    fixed_agent,
    json_output,
    new_case,
    output_equals,
    smelt_agent,
    text,
    text_similar,
    tool,
    tool_call,
)
from smelt.case import SmeltCase
from smelt.given.context import CaseContext, ContextSpec, fresh_workspace
from smelt.results import CaseResult, ExpectationResult
from smelt.runner import _slug, run_case
from smelt.then.expectations import _args_match, _extract_json, _validate_minimal
from smelt.tools import Tool, _schema_for
from smelt.trace import ToolCallRecord, Trace

# ---------------------------------------------------------------------------
# context
# ---------------------------------------------------------------------------


def test_context_spec_merge_combines_all_fields(tmp_path):
    a = context(prompt="A", env={"A": "1"}, vars={"x": 1}, files=tmp_path)
    b = context(prompt="B", env={"B": "2"}, vars={"y": 2})
    merged = a.merge(b)
    assert merged.prompt == "A\n\nB"
    assert merged.env == {"A": "1", "B": "2"}
    assert merged.vars == {"x": 1, "y": 2}
    assert len(merged.files) == 1


def test_context_mapping_files_land_at_destination(tmp_path):
    src = tmp_path / "input.csv"
    src.write_text("a,b", encoding="utf-8")
    result = (
        new_case("mapped")
        .given(context(files={"data/nested.csv": src}))
        .given(fixed_agent("ok"))
        .when(text("go"))
        .then(output_equals("ok"))
        .run()
    )
    assert (Path(result.workspace) / "data" / "nested.csv").read_text() == "a,b"


def test_context_missing_file_raises_error_result():
    result = (
        new_case("missing")
        .given(context(files="definitely/not/here"))
        .given(fixed_agent("ok"))
        .when(text("go"))
        .then(output_equals("ok"))
        .run()
    )
    assert not result.passed and "FileNotFoundError" in result.error


def test_case_context_materialize_merges_env_vars_and_prompt(tmp_path):
    ctx = CaseContext(workspace=tmp_path)
    ctx.materialize([
        context(prompt="one", env={"K": "1"}, vars={"a": 1}),
        context(prompt="two", vars={"b": 2}),
        ContextSpec(),  # an empty spec breaks nothing
    ])
    assert ctx.env == {"K": "1"}
    assert ctx.vars == {"a": 1, "b": 2}
    assert ctx.prompt == "one\n\ntwo"


def test_fresh_workspace_keep_dir_is_used(tmp_path):
    target = fresh_workspace(tmp_path / "keep", name="x")
    assert target == tmp_path / "keep" and target.is_dir()


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------


def test_directory_input_missing_path_raises():
    result = (
        new_case("no-dir")
        .given(fixed_agent("ok"))
        .when(directory("not/a/real/dir"))
        .then(output_equals("ok"))
        .run()
    )
    # fixed_agent never renders the input, so the directory error cannot fire
    assert result.passed

    llm = ScriptedLLM(["ok"])
    result2 = (
        new_case("no-dir-2")
        .given(smelt_agent(llm=llm, system_prompt="s"))
        .when(directory("not/a/real/dir"))
        .then(output_equals("ok"))
        .run()
    )
    assert not result2.passed and "FileNotFoundError" in result2.error


def test_directory_input_custom_note(tmp_path):
    (tmp_path / "f.txt").write_text("x", encoding="utf-8")
    di = directory(tmp_path, note="custom note")
    ws = tmp_path / "ws"
    ws.mkdir()
    rendered = di.render(ws)
    assert rendered.startswith("custom note") and "f.txt" in rendered


# ---------------------------------------------------------------------------
# case
# ---------------------------------------------------------------------------


def test_given_rejects_unknown_type():
    with pytest.raises(TypeError, match="given"):
        new_case("c").given("not a context or agent")


def test_then_rejects_non_expectation():
    case = new_case("c").given(fixed_agent("x")).when(text("go"))
    with pytest.raises(TypeError, match="then"):
        case.then("not an expectation")


def test_case_defaults():
    case = SmeltCase()
    assert case.name == "unnamed" and case.expectations == () and not case.keep_workspace


def test_run_delegates_to_runner():
    result = new_case("r").given(fixed_agent("x")).when(text("go")).run()
    assert isinstance(result, CaseResult) and result.passed


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------


def test_runner_catches_agent_exception():
    class BoomAgent:
        def run(self, ctx, input):
            raise RuntimeError("agent exploded")

    result = run_case(new_case("boom").given(BoomAgent()).when(text("go")))
    assert not result.passed and "agent exploded" in result.error
    assert result.workspace is not None


def test_runner_keep_workspace_under_smelt_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = (
        new_case("Keep Me!", keep_workspace=True)
        .given(fixed_agent("ok"))
        .when(text("go"))
        .then(output_equals("ok"))
        .run()
    )
    assert result.passed
    assert Path(result.workspace) == Path(".smelt") / "Keep-Me"


def test_slug_fallback_and_sanitization():
    assert _slug("a b/c") == "a-b-c"
    assert _slug("！！！") == "case"


# ---------------------------------------------------------------------------
# tools
# ---------------------------------------------------------------------------


def test_tool_schema_inference_with_defaults_and_types():
    def fn(a: str, b: int = 3, c: bool = True, d: list = [], e: dict = {}, f: float = 1.0) -> None:  # noqa: B006
        pass

    schema = _schema_for(fn)
    assert schema["required"] == ["a"]
    assert schema["properties"]["a"] == {"type": "string"}
    assert schema["properties"]["b"] == {"type": "integer"}
    assert schema["properties"]["c"] == {"type": "boolean"}
    assert schema["properties"]["d"] == {"type": "array"}
    assert schema["properties"]["e"] == {"type": "object"}
    assert schema["properties"]["f"] == {"type": "number"}


def test_tool_unannotated_param_gets_empty_schema():
    def fn(mystery) -> None:
        pass

    assert _schema_for(fn)["properties"]["mystery"] == {}


def test_tool_decorator_with_overrides():
    @tool(name="custom", description="custom description")
    def raw_fn(x: str) -> str:
        """Original docstring"""
        return x

    assert raw_fn.name == "custom" and raw_fn.description == "custom description"
    assert raw_fn.invoke({"x": "hi"}) == "hi"
    spec = raw_fn.spec()
    assert spec["name"] == "custom" and spec["parameters"]["type"] == "object"


def test_tool_no_docstring_falls_back_to_fn_name():
    @tool
    def nodoc(x: str) -> str:
        return x

    assert nodoc.description == "nodoc"


def test_tool_explicit_parameters_not_overwritten():
    t = Tool(name="t", description="d", handler=lambda: 1, parameters={"type": "null"})
    assert t.parameters == {"type": "null"}


# ---------------------------------------------------------------------------
# trace / results
# ---------------------------------------------------------------------------


def test_trace_helpers():
    trace = Trace(tool_calls=[
        ToolCallRecord(name="a", arguments={}),
        ToolCallRecord(name="b", arguments={}, error="x"),
        ToolCallRecord(name="a", arguments={"k": 1}),
    ])
    assert trace.called_tools == ["a", "b", "a"]
    assert len(trace.calls_named("a")) == 2
    assert trace.tool_calls[0].ok and not trace.tool_calls[1].ok


def test_case_result_score_edges():
    err = CaseResult(case_name="e", expectations=[], error="x")
    assert err.score == 0.0 and not err.passed

    empty = CaseResult(case_name="ok", expectations=[])
    assert empty.score == 1.0 and empty.passed
    assert "✔" in empty.summary() and "error" not in empty.summary()

    mixed = CaseResult(case_name="m", expectations=[
        ExpectationResult(name="a", score=1.0, threshold=1.0),
        ExpectationResult(name="b", score=0.0, threshold=1.0, message="reason"),
    ])
    assert mixed.score == 0.5 and not mixed.passed
    summary = mixed.summary()
    assert "— reason" in summary and "✘" in summary
    with pytest.raises(AssertionError, match="score=0.50"):
        mixed.assert_passed()
    assert empty.assert_passed() is empty


# ---------------------------------------------------------------------------
# expectations internals
# ---------------------------------------------------------------------------


def test_args_match_nested_and_missing():
    assert _args_match({"a": {"b": 1, "c": 2}}, {"a": {"b": 1}})
    assert not _args_match({"a": {"b": 1}}, {"a": {"b": 2}})
    assert not _args_match({"a": 1}, {"missing": 1})
    assert not _args_match({"a": {"b": 1}}, {"a": {"b": {"c": 1}}})  # asymmetric types


def test_extract_json_all_branches():
    assert _extract_json('[1, 2]') == [1, 2]
    assert _extract_json('prefix {"a": 1} suffix') == {"a": 1}
    assert _extract_json('prefix [3] suffix') == [3]
    assert _extract_json('```\n{"b": 2}\n```') == {"b": 2}
    assert _extract_json('```json\nbroken {json\n```\n{"c": 3}') == {"c": 3}
    with pytest.raises(ValueError, match="no parseable JSON"):
        _extract_json("no {json and no closure")


def test_validate_minimal_types_enum_nested_items():
    schema = {
        "type": "object",
        "required": ["name"],
        "properties": {
            "name": {"type": "string"},
            "level": {"type": "integer"},
            "kind": {"enum": ["a", "b"]},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
    }
    assert _validate_minimal({"name": "x", "level": 1, "kind": "a", "tags": ["t"]}, schema) == []

    errors = _validate_minimal({"level": True, "kind": "z", "tags": ["ok", 5]}, schema)
    assert any("missing required field" in e for e in errors)
    assert any("level" in e and "integer" in e for e in errors)  # bool is not an integer
    assert any("enum" in e for e in errors)
    assert any("tags[1]" in e for e in errors)

    assert _validate_minimal({"level": 3}, {"properties": {"level": {"type": "integer"}}}) == []
    assert _validate_minimal("not-a-dict", {"type": "object"})[0].startswith("$: expected type")
    assert _validate_minimal(None, {"type": "null"}) == []


def test_validate_schema_uses_jsonschema_when_available():
    import jsonschema  # noqa: F401 - installed via dev extras

    from smelt.then.expectations import _validate_schema

    schema = {"type": "object", "properties": {"n": {"type": "integer", "minimum": 5}}}
    assert _validate_schema({"n": 10}, schema) == []
    errors = _validate_schema({"n": 1}, schema)
    assert errors and "minimum" in errors[0] or "less than" in errors[0]


def test_json_output_partial_scoring_weights():
    # parseable (0.4) but schema fails (0.4) and contains passes (0.2) → 0.6
    exp = json_output({"type": "object", "required": ["missing"]}, contains={"a": 1})
    trace = Trace(output='{"a": 1}')
    r = exp.evaluate(trace)
    assert r.score == pytest.approx(0.6) and "required" in r.message
    # a relaxed threshold passes
    exp2 = json_output({"type": "object", "required": ["missing"]}, contains={"a": 1}, threshold=0.5)
    assert exp2.evaluate(trace).passed


def test_output_equals_without_strip():
    from smelt import output_equals as oe

    assert not oe("x", strip=False).evaluate(Trace(output="x ")).passed
    assert oe("x", strip=False).evaluate(Trace(output="x")).passed


def test_text_similar_custom_and_default_name():
    exp = text_similar("reference", threshold=0.5)
    assert "text_similar" in exp.name and "reference" in exp.name
    r = exp.evaluate(Trace(output="something else entirely"))
    assert r.score < 0.5 and "similarity" in r.message


def test_tool_call_expectation_name_property():
    assert "args=" in tool_call("t", args={"a": 1}).name
    assert "args=" not in tool_call("t").name


# ---------------------------------------------------------------------------
# llm.py
# ---------------------------------------------------------------------------


def test_llm_response_helpers():
    assert LLMResponse.say("hi").content == "hi"
    call = LLMResponse.call("t", {"a": 1}).tool_calls[0]
    assert call == ToolCall(name="t", arguments={"a": 1})
    assert LLMResponse.call("t").tool_calls[0].arguments == {}


def test_scripted_llm_repeats_last_and_records_calls():
    llm = ScriptedLLM(["one", LLMResponse.say("two")])
    assert llm.complete([{"role": "user", "content": "1"}], []).content == "one"
    assert llm.complete([], []).content == "two"
    assert llm.complete([], []).content == "two"  # repeats the last once exhausted
    assert len(llm.calls) == 3


def test_scripted_llm_rejects_empty():
    with pytest.raises(ValueError, match="at least one response"):
        ScriptedLLM([])


def test_openai_client_with_fake_module(monkeypatch):
    captured = {}

    class FakeFunction:
        name = "run_command"
        arguments = '{"cmd": "ls"}'

    fake_message = types.SimpleNamespace(
        content="done",
        tool_calls=[types.SimpleNamespace(function=FakeFunction())],
    )

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=fake_message)])

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured["client_kwargs"] = kwargs
            self.chat = types.SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))

    from smelt.given.agents.llm import OpenAIChatClient

    @tool
    def run_command(cmd: str) -> str:
        """Run a command"""
        return "ok"

    client = OpenAIChatClient("kimi-k2", base_url="https://example.com/v1", api_key="sk-x", temperature=0.3)
    assert captured["client_kwargs"] == {"base_url": "https://example.com/v1", "api_key": "sk-x"}

    resp = client.complete([{"role": "user", "content": "hi"}], [run_command])
    assert captured["model"] == "kimi-k2" and captured["temperature"] == 0.3
    assert captured["tool_choice"] == "auto"
    assert captured["tools"][0]["function"]["name"] == "run_command"
    assert resp.content == "done"
    assert resp.tool_calls[0] == ToolCall(name="run_command", arguments={"cmd": "ls"})

    # without tools, the request carries no tools fields
    captured.clear()
    client.complete([], [])
    assert "tools" not in captured and "tool_choice" not in captured


def test_openai_client_import_error_message(monkeypatch):
    monkeypatch.setitem(sys.modules, "openai", None)
    from smelt.given.agents.llm import OpenAIChatClient

    with pytest.raises(ImportError, match="smelt\\[openai\\]"):
        OpenAIChatClient("any")


def test_openai_client_extra_body_passthrough(monkeypatch):
    """Provider-specific switches (e.g. thinking-mode disable) reach the request body."""
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="ok", tool_calls=None))],
            )

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = types.SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))

    from smelt.given.agents.llm import OpenAIChatClient

    client = OpenAIChatClient("m", api_key="sk-x", extra_body={"thinking": {"type": "disabled"}})
    client.complete([{"role": "user", "content": "hi"}], [])
    assert captured["extra_body"] == {"thinking": {"type": "disabled"}}

    captured.clear()
    OpenAIChatClient("m", api_key="sk-x").complete([], [])
    assert "extra_body" not in captured  # omitted by default


# ---------------------------------------------------------------------------
# smelt agent edges
# ---------------------------------------------------------------------------


def test_smelt_agent_requires_llm():
    from smelt.given.agents.smelt import SmeltAgent

    with pytest.raises(ValueError, match="requires an llm"):
        SmeltAgent().run(CaseContext(workspace=Path(".")), text("hi"))


def test_smelt_agent_max_turns_terminates():
    llm = ScriptedLLM([LLMResponse.call("echo")])  # always requests a tool call

    @tool
    def echo() -> str:
        """Echo"""
        return "x"

    result = (
        new_case("loop")
        .given(smelt_agent(llm=llm, tools=[echo], max_turns=3, system_prompt="s"))
        .when(text("go"))
        .then(output_equals("(max_turns reached without a final answer)"))
        .run()
    )
    assert result.passed
    assert len(result.trace.tool_calls) == 3


def test_smelt_agent_tool_exception_recorded_in_trace():
    llm = ScriptedLLM([LLMResponse.call("explode"), LLMResponse.say("error handled")])

    @tool
    def explode() -> str:
        """Blows up"""
        raise RuntimeError("boom")

    result = (
        new_case("explode")
        .given(smelt_agent(llm=llm, tools=[explode], system_prompt="s"))
        .when(text("go"))
        .then(tool_call("explode"))
        .then(output_equals("error handled"))
        .run()
    )
    assert result.passed
    record = result.trace.tool_calls[0]
    assert record.error == "RuntimeError: boom" and not record.ok
    # the error is fed back to the LLM as a tool message
    tool_msgs = [m for m in result.trace.messages if m.get("role") == "tool"]
    assert "boom" in tool_msgs[0]["content"]


def test_smelt_agent_env_injected_and_restored(tmp_path):
    seen = {}

    @tool
    def read_env() -> str:
        """Read an env var"""
        seen["during"] = os.environ.get("SMELT_CASE_ENV")
        return "ok"

    os.environ.pop("SMELT_CASE_ENV", None)
    llm = ScriptedLLM([LLMResponse.call("read_env"), LLMResponse.say("done")])
    result = (
        new_case("env")
        .given(context(env={"SMELT_CASE_ENV": "injected"}))
        .given(smelt_agent(llm=llm, tools=[read_env], system_prompt="s"))
        .when(text("go"))
        .then(output_equals("done"))
        .run()
    )
    assert result.passed
    assert seen["during"] == "injected"
    assert os.environ.get("SMELT_CASE_ENV") is None  # restored afterwards


def test_smelt_agent_loads_skill_variants(tmp_path):
    from smelt.given.agents.smelt import _load_skill_prompt

    skill_dir = tmp_path / "my_skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: x\n---\n\n# Body content\n", encoding="utf-8"
    )
    assert _load_skill_prompt(skill_dir) == "# Body content"  # directory → SKILL.md, frontmatter stripped
    assert _load_skill_prompt(skill_dir / "SKILL.md") == "# Body content"

    plain = tmp_path / "plain.md"
    plain.write_text("no frontmatter", encoding="utf-8")
    assert _load_skill_prompt(plain) == "no frontmatter"

    with pytest.raises(FileNotFoundError, match="skill not found"):
        _load_skill_prompt(tmp_path / "ghost")


def test_smelt_agent_system_prompt_composition(tmp_path):
    from smelt.given.agents.smelt import SmeltAgent

    skill = tmp_path / "SKILL.md"
    skill.write_text("---\nname: s\n---\nskill body", encoding="utf-8")
    agent = SmeltAgent(skill=skill, system_prompt="direct prompt")
    ctx = CaseContext(workspace=tmp_path, prompt="given context")
    composed = agent._build_system_prompt(ctx)
    assert "direct prompt" in composed and "skill body" in composed and "given context" in composed


def test_fixed_agent_metadata_passthrough():
    from smelt.given.agents.fixed import FixedAgent

    agent = FixedAgent(output="x", metadata={"model": "v1"})
    trace = agent.run(CaseContext(workspace=Path(".")), text("hi"))
    assert trace.metadata == {"model": "v1"}


# ---------------------------------------------------------------------------
# pytest fixture (the plugin entry point ships with the package)
# ---------------------------------------------------------------------------


def test_smelt_fixture_available(smelt):
    case = new_case("via-fixture").given(fixed_agent("ok")).when(text("go")).then(output_equals("ok"))
    assert smelt.check(case).passed
    assert len(smelt.results) == 1


# ---------------------------------------------------------------------------
# CLI edges
# ---------------------------------------------------------------------------


def test_cli_run_unloadable_file_returns_2(tmp_path, capsys):
    bad = tmp_path / "bad.py"
    bad.write_text("raise RuntimeError('explodes on import')", encoding="utf-8")
    from smelt.cli import main

    assert main(["run", str(bad)]) == 2
    assert "failed to load" in capsys.readouterr().err


def test_cli_run_no_cases_returns_2(tmp_path, capsys):
    empty = tmp_path / "empty.py"
    empty.write_text("X = 1", encoding="utf-8")
    from smelt.cli import main

    assert main(["run", str(empty)]) == 2
    assert "no SmeltCase found" in capsys.readouterr().err


def test_cli_run_verbose_prints_passing_details(tmp_path, capsys):
    f = tmp_path / "c.py"
    f.write_text(
        "from smelt import new_case, fixed_agent, text, output_equals\n"
        "cases = [new_case('v').given(fixed_agent('d')).when(text('g')).then(output_equals('d'))]\n",
        encoding="utf-8",
    )
    from smelt.cli import main

    assert main(["run", str(f), "-v"]) == 0
    out = capsys.readouterr().out
    assert "output_equals" in out and "1/1 cases passed" in out
