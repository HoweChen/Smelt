"""Tests for budget assertions (operating envelopes): turns / tool calls / wall time.

Budgets are deterministic gates on the trace — quality can be fine while the
run is unaffordable ("budget burner"); these assertions catch that without any
LLM call.
"""

import time

from smelt import (
    LLMResponse,
    ScriptedLLM,
    fixed_agent,
    new_case,
    smelt_agent,
    text,
    tool,
    tool_budget,
    turns_used,
    wall_time,
)
from smelt.trace import Trace


@tool
def run_command(cmd: str) -> str:
    """Run a command"""
    return "ok"


@tool
def read_file(path: str) -> str:
    """Read a file"""
    return "content"


def _agent_case(agent, *expectations):
    return new_case("budget").given(agent).when(text("go")).then(*expectations)


# ---------------------------------------------------------------------------
# turns_used
# ---------------------------------------------------------------------------


def _three_turn_agent():
    # call → call → say: three LLM turns
    return smelt_agent(
        llm=ScriptedLLM([
            LLMResponse.call("run_command", {"cmd": "ls"}),
            LLMResponse.call("read_file", {"path": "a.txt"}),
            LLMResponse.say("done"),
        ]),
        system_prompt="test",
        tools=[run_command, read_file],
    )


def test_turns_used_within_budget_passes():
    result = _agent_case(_three_turn_agent(), turns_used(max=5)).run()
    e = result.expectations[0]
    assert e.passed and e.score == 1.0
    assert result.trace.turns == 3


def test_turns_used_over_budget_fails_with_counts():
    result = _agent_case(_three_turn_agent(), turns_used(max=2)).run()
    e = result.expectations[0]
    assert not e.passed and e.score == 0.0
    assert "3" in e.message and "2" in e.message


def test_turns_used_fixed_agent_counts_one_turn():
    result = _agent_case(fixed_agent("done"), turns_used(max=1)).run()
    assert result.expectations[0].passed
    assert result.trace.turns == 1


# ---------------------------------------------------------------------------
# tool_budget
# ---------------------------------------------------------------------------


def _chatty_agent():
    return fixed_agent("done", tool_calls=[
        {"name": "run_command", "arguments": {"cmd": "a"}},
        {"name": "run_command", "arguments": {"cmd": "b"}},
        {"name": "run_command", "arguments": {"cmd": "c"}},
        {"name": "read_file", "arguments": {"path": "x"}},
    ])


def test_tool_budget_per_tool():
    over = _agent_case(_chatty_agent(), tool_budget("run_command", max=2)).run()
    assert not over.expectations[0].passed
    assert "3" in over.expectations[0].message

    ok = _agent_case(_chatty_agent(), tool_budget("run_command", max=3)).run()
    assert ok.expectations[0].passed


def test_tool_budget_total_when_no_name_given():
    over = _agent_case(_chatty_agent(), tool_budget(max=3)).run()
    assert not over.expectations[0].passed  # 4 total calls

    ok = _agent_case(_chatty_agent(), tool_budget(max=4)).run()
    assert ok.expectations[0].passed


def test_tool_budget_validates():
    import pytest

    with pytest.raises(ValueError, match=">= 0"):
        tool_budget(max=-1)


# ---------------------------------------------------------------------------
# wall_time + trace timing
# ---------------------------------------------------------------------------


class SlowAgent:
    def run(self, ctx, input):
        time.sleep(0.05)
        return Trace(output="done", turns=1)


def test_wall_time_recorded_and_budget_enforced():
    over = _agent_case(SlowAgent(), wall_time(max_seconds=0.01)).run()
    e = over.expectations[0]
    assert not e.passed and e.score == 0.0
    assert over.trace.wall_time_s is not None and over.trace.wall_time_s >= 0.05

    ok = _agent_case(SlowAgent(), wall_time(max_seconds=10)).run()
    assert ok.expectations[0].passed


def test_wall_time_unrecorded_passes_with_note():
    # a manually built trace without timing cannot violate a time budget
    from smelt.then.expectations import WallTimeExpectation

    e = WallTimeExpectation(max_seconds=0.001).evaluate(Trace(output="x"))
    assert e.passed and "no timing" in e.message


# ---------------------------------------------------------------------------
# token usage recording (optional, recorded when the client reports it)
# ---------------------------------------------------------------------------


def test_token_usage_accumulated_into_trace_metadata():
    class MeteredLLM:
        def __init__(self):
            self._responses = [
                LLMResponse.call("run_command", {"cmd": "ls"}),
                LLMResponse.say("done"),
            ]

        def complete(self, messages, tools):
            r = self._responses.pop(0)
            return LLMResponse(
                content=r.content,
                tool_calls=r.tool_calls,
                usage={"prompt_tokens": 100, "completion_tokens": 20},
            )

    agent = smelt_agent(llm=MeteredLLM(), system_prompt="t", tools=[run_command])
    result = _agent_case(agent, turns_used(max=5)).run()
    usage = result.trace.metadata["token_usage"]
    assert usage == {"prompt_tokens": 200, "completion_tokens": 40}


def test_no_usage_reported_no_metadata():
    agent = smelt_agent(llm=ScriptedLLM([LLMResponse.say("done")]), system_prompt="t")
    result = _agent_case(agent, turns_used(max=5)).run()
    assert "token_usage" not in result.trace.metadata
