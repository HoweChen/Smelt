"""Runner guards: missing/conflicting DSL segments must fail safely with guidance.

Matrix:
- no given (no agent, no fragments) but when/then → cannot run
- given with only a context (no agent) → cannot run
- fragmented given without llm → cannot run
- no when → cannot run
- no then → runs fine (empty assertions count as a pass, score 1.0)
- agent exploding at run time → converges to error, never raised
- build-time DSL conflicts (double agent, double when, wrong types) → raise immediately
"""

import pytest

from smelt import (
    LLMResponse,
    ScriptedLLM,
    context,
    fixed_agent,
    llm,
    new_case,
    output_equals,
    skill,
    smelt_agent,
    text,
    tool,
    tools,
)
from smelt.results import CaseResult


@tool
def noop() -> str:
    """No-op"""
    return "ok"


# ---------------------------------------------------------------------------
# Missing given: cannot run
# ---------------------------------------------------------------------------


def test_no_given_at_all_cannot_run():
    """No given at all, straight to when + then — the runner refuses to execute."""
    result = new_case("no-given").when(text("go")).then(output_equals("x")).run()
    assert isinstance(result, CaseResult)
    assert not result.passed
    assert "missing agent" in result.error


def test_given_only_context_cannot_run():
    """A context was given but no agent — still cannot run."""
    result = (
        new_case("context-only")
        .given(context(prompt="background only, no agent"))
        .when(text("go"))
        .then(output_equals("x"))
        .run()
    )
    assert not result.passed and "missing agent" in result.error


def test_given_fragments_without_llm_cannot_run():
    """Fragmented given without llm — assembly fails, cannot run."""
    result = (
        smelt_agent.new_case("no-llm")
        .given(skill(prompt="a prompt"))
        .given(tools(noop))
        .when(text("go"))
        .then(output_equals("x"))
        .run()
    )
    assert not result.passed and "missing llm fragment" in result.error


def test_given_only_llm_fragment_runs():
    """Fragmented given with just an llm works (skill and tools are optional)."""
    result = (
        smelt_agent.new_case("llm-only")
        .given(llm(ScriptedLLM(["ok"])))
        .when(text("go"))
        .then(output_equals("ok"))
        .run()
    )
    assert result.passed, result.summary()


# ---------------------------------------------------------------------------
# Missing when: cannot run
# ---------------------------------------------------------------------------


def test_no_when_cannot_run():
    result = new_case("no-when").given(fixed_agent("x")).then(output_equals("x")).run()
    assert not result.passed and "missing trigger" in result.error


def test_no_when_with_fragments_cannot_run():
    result = (
        smelt_agent.new_case("no-when-frag")
        .given(llm(ScriptedLLM(["ok"])))
        .then(output_equals("ok"))
        .run()
    )
    assert not result.passed and "missing trigger" in result.error


def test_nothing_at_all_reports_agent_error_first():
    """With nothing at all, the missing-agent error is reported first
    (the agent is the prerequisite for execution)."""
    result = new_case("empty").run()
    assert not result.passed and "missing agent" in result.error


# ---------------------------------------------------------------------------
# Missing then: runs (empty assertions = flow verification)
# ---------------------------------------------------------------------------


def test_no_then_runs_and_passes_vacuously():
    """then may be omitted: verifies only that the flow runs; empty assertions
    count as a pass with score 1.0."""
    result = new_case("no-then").given(fixed_agent("ok")).when(text("go")).run()
    assert result.passed and result.score == 1.0 and result.expectations == []


# ---------------------------------------------------------------------------
# Runtime exceptions: converge to error, never raised
# ---------------------------------------------------------------------------


def test_agent_exception_converges_to_error_result():
    class BoomAgent:
        def run(self, ctx, input):
            raise RuntimeError("boom")

    result = new_case("boom").given(BoomAgent()).when(text("go")).then(output_equals("x")).run()
    assert not result.passed and "boom" in result.error


def test_llm_infinite_tool_loop_bounded_by_max_turns():
    """An LLM forever requesting tool calls is capped by max_turns instead of looping."""
    llm_client = ScriptedLLM([LLMResponse.call("noop")])  # repeats the last response forever
    result = (
        smelt_agent.new_case("loop")
        .given(llm(llm_client))
        .given(tools(noop))
        .when(text("go"))
        .run()
    )
    assert result.passed  # it terminates
    assert "max_turns" in result.trace.output
    assert len(result.trace.tool_calls) == 8  # default cap


# ---------------------------------------------------------------------------
# Build-time conflicts: raise immediately (before run)
# ---------------------------------------------------------------------------


def test_double_agent_rejected_at_build_time():
    case = new_case("c").given(fixed_agent("a"))
    with pytest.raises(ValueError, match="only one agent per case"):
        case.given(fixed_agent("b"))


def test_double_when_rejected_at_build_time():
    case = new_case("c").given(fixed_agent("a")).when(text("x"))
    with pytest.raises(ValueError, match="only one trigger per case"):
        case.when(text("y"))


def test_given_rejects_unknown_type_at_build_time():
    with pytest.raises(TypeError, match="given"):
        new_case("c").given("whatever this is")


def test_then_rejects_non_expectation_at_build_time():
    case = new_case("c").given(fixed_agent("a")).when(text("x"))
    with pytest.raises(TypeError, match="then"):
        case.then("whatever this is")
