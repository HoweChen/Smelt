"""Tests for global default thresholds + fragmented given (skill/llm/tools) +
the smelt_agent.new_case entry point."""


import pytest

import smelt
from smelt import (
    LLMResponse,
    ScriptedLLM,
    SmeltAgent,
    context,
    llm,
    llm_judge,
    new_case,
    output_equals,
    skill,
    smelt_agent,
    text,
    text_similar,
    tool,
    tool_call,
    tools,
)


@pytest.fixture(autouse=True)
def reset_config():
    yield
    smelt.configure(text_similar_threshold=0.8, llm_judge_threshold=0.8)  # restore global defaults


# ---------------------------------------------------------------------------
# Default thresholds
# ---------------------------------------------------------------------------


def test_text_similar_default_threshold_is_0_8():
    exp = text_similar("reference")
    assert exp.threshold == 0.8


def test_llm_judge_default_threshold_is_0_8():
    exp = llm_judge(ScriptedLLM(["{}"]), criteria="x")
    assert exp.threshold == 0.8


def test_configure_changes_global_defaults():
    smelt.configure(text_similar_threshold=0.6, llm_judge_threshold=0.7)
    assert text_similar("reference").threshold == 0.6
    assert llm_judge(ScriptedLLM(["{}"]), criteria="x").threshold == 0.7


def test_explicit_threshold_always_wins():
    smelt.configure(text_similar_threshold=0.3)
    assert text_similar("reference", threshold=0.95).threshold == 0.95


def test_configure_validates_range():
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        smelt.configure(llm_judge_threshold=1.5)
    # a failed validation leaves the config untouched
    assert smelt.config.llm_judge_threshold == 0.8


def test_default_threshold_end_to_end():
    smelt.configure(text_similar_threshold=0.2)
    # output and reference differ a lot (similarity 0.25): fails at the 0.8 default,
    # passes once the default drops to 0.2
    result = (
        new_case("thresh")
        .given(smelt.fixed_agent("a partially related answer"))
        .when(text("go"))
        .then(text_similar("a completely different reference"))
        .run()
    )
    assert result.expectations[0].score < 0.8
    assert result.passed


# ---------------------------------------------------------------------------
# Fragmented given + the smelt_agent.new_case entry
# ---------------------------------------------------------------------------


@tool
def echo(text: str) -> str:
    """Echo"""
    return text.upper()


def test_agent_first_entry_with_fragments(tmp_path):
    skill_dir = tmp_path / "s"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: s\ndescription: d\n---\n# Skill body\n", encoding="utf-8")

    agent_llm = ScriptedLLM([
        LLMResponse.call("echo", {"text": "hi"}),
        LLMResponse.say("HI echoed"),
    ])
    result = (
        smelt_agent.new_case("agent-first")
        .given(skill(skill_dir))
        .given(llm(agent_llm))
        .given(tools(echo))
        .when(text("echo hi"))
        .then(tool_call("echo", args={"text": "hi"}))
        .then(output_equals("HI echoed"))
        .run()
    )
    assert result.passed, result.summary()
    # the skill document really made it into the system prompt
    system_msg = result.trace.messages[0]["content"]
    assert "Skill body" in system_msg


def test_smelt_agent_alias_is_class_and_new_case_returns_smelt_case():
    assert smelt_agent is SmeltAgent
    case = smelt_agent.new_case("x")
    assert isinstance(case, smelt.SmeltCase)


def test_fragments_with_prompt_instead_of_skill_file():
    agent_llm = ScriptedLLM(["done"])
    result = (
        smelt_agent.new_case("prompt-skill")
        .given(skill(prompt="You are an assistant"))
        .given(llm(agent_llm))
        .when(text("go"))
        .then(output_equals("done"))
        .run()
    )
    assert result.passed
    assert "You are an assistant" in result.trace.messages[0]["content"]


def test_tools_fragments_accumulate():
    @tool
    def another() -> str:
        """Another one"""
        return "x"

    case = (
        smelt_agent.new_case("multi-tools")
        .given(tools(echo))
        .given(tools(another))
        .given(llm(ScriptedLLM(["ok"])))
        .when(text("go"))
    )
    # assembly happens at run time; verify the assembly logic directly here
    from smelt.given.fragments import assemble_agent

    agent = assemble_agent(case.fragments)
    assert {t.name for t in agent.tools} == {"echo", "another"}


def test_fragments_mix_with_context():
    agent_llm = ScriptedLLM(["ok"])
    result = (
        smelt_agent.new_case("mixed")
        .given(context(prompt="background info"))
        .given(llm(agent_llm))
        .when(text("go"))
        .then(output_equals("ok"))
        .run()
    )
    assert result.passed
    assert "background info" in result.trace.messages[0]["content"]


def test_missing_llm_fragment_converges_to_error():
    result = (
        smelt_agent.new_case("no-llm")
        .given(tools(echo))
        .when(text("go"))
        .then(output_equals("x"))
        .run()
    )
    assert not result.passed and "missing llm fragment" in result.error


def test_duplicate_skill_or_llm_fragment_rejected():
    case = (
        smelt_agent.new_case("dup")
        .given(skill(prompt="a"))
        .given(skill(prompt="b"))
        .given(llm(ScriptedLLM(["x"])))
        .when(text("go"))
    )
    result = case.run()
    assert not result.passed and "duplicate skill fragment" in result.error

    case2 = (
        smelt_agent.new_case("dup2")
        .given(llm(ScriptedLLM(["x"])))
        .given(llm(ScriptedLLM(["y"])))
        .when(text("go"))
    )
    assert "duplicate llm fragment" in case2.run().error


def test_fragments_and_explicit_agent_are_exclusive():
    # an explicit agent wins; fragments are ignored (no error, no conflict)
    result = (
        smelt_agent.new_case("both")
        .given(smelt.fixed_agent("fixed wins"))
        .given(tools(echo))  # fragment present but an explicit agent is set
        .when(text("go"))
        .then(output_equals("fixed wins"))
        .run()
    )
    assert result.passed


def test_skill_requires_source_or_prompt():
    with pytest.raises(ValueError, match="at least one"):
        skill()


def test_tools_rejects_non_tool():
    with pytest.raises(TypeError, match="@tool"):
        tools(lambda: 1)


def test_given_still_rejects_unknown_types():
    with pytest.raises(TypeError, match="given"):
        smelt_agent.new_case("c").given(42)
