"""SmeltAgent: smelt's built-in LLM tool-loop agent that loads a SKILL.md.

System prompt = skill document + given contexts; then runs the
"LLM → tool execution → feed result back" loop until the LLM returns plain text
or max_turns is hit. Everything is recorded into the Trace.
"""

from __future__ import annotations

import itertools
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from smelt.given.agents.llm import LLMClient
from smelt.given.context import CaseContext
from smelt.tools import Tool
from smelt.trace import ToolCallRecord, Trace
from smelt.when.inputs import CaseInput


def _load_skill_prompt(skill: str | os.PathLike[str]) -> str:
    """Read a SKILL.md (file or containing directory), strip frontmatter, use as system prompt."""
    path = Path(skill)
    if path.is_dir():
        path = path / "SKILL.md"
    if not path.exists():
        raise FileNotFoundError(f"skill not found: {skill}")
    body = path.read_text(encoding="utf-8")
    if body.startswith("---"):
        parts = body.split("---", 2)
        if len(parts) == 3:
            body = parts[2]
    return body.strip()


@dataclass
class SmeltAgent:
    """An LLM agent that loads a skill.

    - ``skill``: SKILL.md path or its directory; content goes into the system prompt;
    - ``llm``: an LLMClient — ScriptedLLM in tests, a real model for regression;
    - ``tools``: tools the agent may call; handlers run inside the workspace;
    - ``max_turns``: tool-loop cap, prevents infinite loops.
    """

    skill: str | os.PathLike[str] | None = None
    llm: LLMClient | None = None
    tools: Sequence[Tool] = ()
    max_turns: int = 8
    system_prompt: str | None = None  # direct prompt when not using a skill file

    @classmethod
    def new_case(cls, name: str = "unnamed", *, keep_workspace: bool = False):
        """Agent-first case entry: smelt_agent.new_case("x").given(skill(...)).given(llm(...))…"""
        from smelt.case import new_case

        return new_case(name, keep_workspace=keep_workspace)

    def run(self, ctx: CaseContext, input: CaseInput) -> Trace:
        if self.llm is None:
            raise ValueError("SmeltAgent requires an llm (LLMClient), or use fixed_agent for fixed output")

        user_text = input.render(ctx.workspace)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._build_system_prompt(ctx)},
            {"role": "user", "content": user_text},
        ]
        trace = Trace(messages=messages)

        cwd = Path.cwd()
        os.chdir(ctx.workspace)
        turns = 0
        usage_total: dict[str, int] = {}
        id_counter = itertools.count(1)
        try:
            for _ in range(self.max_turns):
                response = self.llm.complete(messages, self.tools)
                turns += 1
                if response.usage:
                    for key, value in response.usage.items():
                        usage_total[key] = usage_total.get(key, 0) + int(value)
                if not response.tool_calls:
                    trace.output = response.content
                    trace.turns = turns
                    if usage_total:
                        trace.metadata["token_usage"] = usage_total
                    messages.append({"role": "assistant", "content": response.content})
                    return trace

                # OpenAI wire format: one assistant message carries all tool calls,
                # each with an id; every tool result pairs back via tool_call_id.
                # Strict providers (DeepSeek et al.) 422 on anything less.
                calls = response.tool_calls
                ids = [f"smelt-call-{next(id_counter)}" for _ in calls]
                messages.append({
                    "role": "assistant",
                    "content": response.content,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(call.arguments, ensure_ascii=False, default=str),
                            },
                        }
                        for call, call_id in zip(calls, ids)
                    ],
                })
                for call, call_id in zip(calls, ids):
                    record = self._execute(call.name, call.arguments, ctx)
                    trace.tool_calls.append(record)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": record.name,
                        "content": record.error if record.error else json.dumps(record.result, ensure_ascii=False, default=str),
                    })
            trace.output = "(max_turns reached without a final answer)"
            trace.turns = turns
            if usage_total:
                trace.metadata["token_usage"] = usage_total
            return trace
        finally:
            os.chdir(cwd)

    def _build_system_prompt(self, ctx: CaseContext) -> str:
        parts = []
        if self.system_prompt:
            parts.append(self.system_prompt)
        if self.skill is not None:
            parts.append(_load_skill_prompt(self.skill))
        if ctx.prompt:
            parts.append(f"## Context\n{ctx.prompt}")
        return "\n\n".join(parts)

    def _execute(self, name: str, arguments: dict[str, Any], ctx: CaseContext) -> ToolCallRecord:
        tool = next((t for t in self.tools if t.name == name), None)
        if tool is None:
            return ToolCallRecord(name=name, arguments=arguments, error=f"unknown tool: {name}")
        old_env = {k: os.environ.get(k) for k in ctx.env}
        os.environ.update(ctx.env)
        try:
            return ToolCallRecord(name=name, arguments=arguments, result=tool.invoke(arguments))
        except Exception as e:  # noqa: BLE001 - tool errors go into the trace instead of crashing the case
            return ToolCallRecord(name=name, arguments=arguments, error=f"{type(e).__name__}: {e}")
        finally:
            for k, v in old_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


# smelt_agent is both the entry point (smelt_agent.new_case(...))
# and the constructor (given(smelt_agent(skill_path, llm=..., tools=[...]))).
smelt_agent = SmeltAgent
