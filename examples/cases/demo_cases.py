"""Example case file that `smelt run` can execute directly.

Usage: smelt run examples/cases/demo_cases.py -v
"""

from smelt import (
    LLMResponse,
    ScriptedLLM,
    fixed_agent,
    json_output,
    new_case,
    output_contains,
    smelt_agent,
    text,
    tool,
    tool_call,
)


@tool
def run_command(cmd: str) -> str:
    """Run a shell command in the workspace (not actually executed in this demo)"""
    return f"executed: {cmd}"


# 1) Fixed-output path: replay a known trace to verify assertions and baselines
baseline_commit = (
    new_case("commit-baseline")
    .given(fixed_agent(
        "committed as abc123",
        tool_calls=[{"name": "run_command", "arguments": {"cmd": "git commit -m update"}}],
    ))
    .when(text("commit my changes"))
    .then(tool_call("run_command"))
    .then(output_contains("committed"))
)

# 2) LLM as agent: ScriptedLLM simulates the model's tool-loop decisions
agent_loop = (
    new_case("agent-loop")
    .given(smelt_agent(
        llm=ScriptedLLM([
            LLMResponse.call("run_command", {"cmd": "git status"}),
            LLMResponse.call("run_command", {"cmd": "git commit -m update"}),
            LLMResponse.say("status checked and changes committed"),
        ]),
        tools=[run_command],
        system_prompt="You are a git assistant",
    ))
    .when(text("commit my changes"))
    .then(tool_call("run_command", args={"cmd": "git status"}))
    .then(tool_call("run_command", args={"cmd": "git commit -m update"}))
    .then(output_contains("checked", "committed"))
)

# 3) JSON output validation
json_case = (
    new_case("json-output")
    .given(fixed_agent('{"skill": "commit", "score": 92}'))
    .when(text("report as JSON"))
    .then(json_output(
        {"type": "object", "required": ["skill", "score"]},
        contains={"skill": "commit"},
    ))
)

cases = [baseline_commit, agent_loop, json_case]
