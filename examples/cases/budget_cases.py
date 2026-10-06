"""Budget assertions (operating envelopes) — same task, two cost profiles.

Usage: uv run smelt run examples/cases/budget_cases.py -v

Both agents produce the correct final answer. The efficient one stays within
budget; the wasteful one gets the same result but loops and re-reads files —
the "budget burner" failure mode that quality assertions alone cannot see.

Expected outcome: budget-efficient passes; budget-wasteful FAILS its budget
assertions (exit code 1) — that failure is the demo.
"""

import time

from smelt import (
    LLMResponse,
    ScriptedLLM,
    new_case,
    output_contains,
    smelt_agent,
    text,
    tool,
    tool_budget,
    turns_used,
    wall_time,
)


@tool
def read_file(path: str) -> str:
    """Read a file from the workspace"""
    return f"contents of {path}"


@tool
def write_file(path: str, content: str) -> str:
    """Write a file into the workspace"""
    return f"wrote {path}"


# 1) Efficient agent: read once, write once, report — 3 turns, 2 tool calls
efficient = (
    new_case("budget-efficient")
    .given(smelt_agent(
        llm=ScriptedLLM([
            LLMResponse.call("read_file", {"path": "raw.txt"}),
            LLMResponse.call("write_file", {"path": "notes.md", "content": "# Minutes"}),
            LLMResponse.say("Done — notes.md written."),
        ]),
        system_prompt="You distill meeting notes",
        tools=[read_file, write_file],
    ))
    .when(text("distill raw.txt into minutes"))
    .then(output_contains("notes.md"))
    .then(turns_used(max=4))                 # 3 turns used, within budget
    .then(tool_budget("read_file", max=1))   # read once — enough
    .then(tool_budget(max=3))                # total call cap
    .then(wall_time(max_seconds=10))
)

# 2) Wasteful agent: same answer, but re-reads the file and wanders —
#    quality assertion passes, budget assertions fail
wasteful = (
    new_case("budget-wasteful")
    .given(smelt_agent(
        llm=ScriptedLLM([
            LLMResponse.call("read_file", {"path": "raw.txt"}),
            LLMResponse.call("read_file", {"path": "raw.txt"}),   # redundant re-read
            LLMResponse.call("read_file", {"path": "raw.txt"}),   # ...and again
            LLMResponse.call("write_file", {"path": "draft.md", "content": "# Draft"}),
            LLMResponse.call("write_file", {"path": "notes.md", "content": "# Minutes"}),
            LLMResponse.say("Done — notes.md written."),
        ]),
        system_prompt="You distill meeting notes",
        tools=[read_file, write_file],
    ))
    .when(text("distill raw.txt into minutes"))
    .then(output_contains("notes.md"))       # quality: still correct ✔
    .then(turns_used(max=4))                 # ✘ 6 turns
    .then(tool_budget("read_file", max=1))   # ✘ 3 reads
    .then(tool_budget(max=3))                # ✘ 5 calls total
)


class SlowAgent:
    """Custom agent that takes its time — wall_time budget demo."""

    def run(self, ctx, input):
        time.sleep(0.05)
        from smelt.trace import Trace

        return Trace(output="slow but correct", turns=1)


# 3) Wall-clock budget: correct answer, too slow
slow = (
    new_case("budget-slow")
    .given(SlowAgent())
    .when(text("answer quickly"))
    .then(output_contains("correct"))        # quality: fine ✔
    .then(wall_time(max_seconds=0.01))       # ✘ ~0.05s over the 0.01s budget
)

cases = [efficient, wasteful, slow]
