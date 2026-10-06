"""given fragments: declare skill / llm / tools in segments, assembled into a
SmeltAgent at run time.

Pairs with the agent-first entry style::

    from smelt import smelt_agent, skill, llm, tools

    result = (
        smelt_agent.new_case("commit")
        .given(skill("skills/commit"))
        .given(llm(my_llm))
        .given(tools(read_file, write_file))
        .when(text("commit my changes"))
        .then(tool_call("run_command"))
        .run()
    )

Fragments mix freely with a whole agent: ``.given(smelt_agent(...))`` still works.
An explicit agent takes precedence — once one is given, fragments are ignored.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from smelt.given.agents.llm import LLMClient
from smelt.given.agents.smelt import SmeltAgent
from smelt.tools import Tool


@dataclass(frozen=True)
class SkillSpec:
    """skill fragment: a SKILL.md path/directory, or a direct system prompt."""

    source: Path | None = None
    prompt: str | None = None


@dataclass(frozen=True)
class LLMSpec:
    """llm fragment: any LLMClient (ScriptedLLM / OpenAIChatClient / LangChainLLM)."""

    client: LLMClient


@dataclass(frozen=True)
class ToolsSpec:
    """tools fragment: callable tools; multiple tools() fragments accumulate."""

    items: tuple[Tool, ...] = field(default_factory=tuple)


AgentFragment = SkillSpec | LLMSpec | ToolsSpec


def skill(source: str | os.PathLike[str] | None = None, *, prompt: str | None = None) -> SkillSpec:
    """given(skill("skills/commit")) or given(skill(prompt="You are..."))."""
    if source is None and prompt is None:
        raise ValueError("skill() needs at least one of source path or prompt")
    return SkillSpec(source=Path(source) if source is not None else None, prompt=prompt)


def llm(client: LLMClient) -> LLMSpec:
    """given(llm(my_client))."""
    return LLMSpec(client=client)


def tools(*items: Tool) -> ToolsSpec:
    """given(tools(read_file, write_file))."""
    for t in items:
        if not isinstance(t, Tool):
            raise TypeError(f"tools() only accepts @tool-decorated functions, got: {type(t).__name__}")
    return ToolsSpec(items=tuple(items))


def assemble_agent(fragments: Sequence[AgentFragment], *, max_turns: int = 8) -> SmeltAgent:
    """Assemble fragments into a SmeltAgent.

    - at most one skill fragment (source and prompt live in the same fragment);
    - at most one llm fragment, and it is required;
    - tools fragments may repeat and accumulate.
    """
    skill_spec: SkillSpec | None = None
    llm_spec: LLMSpec | None = None
    tool_items: list[Tool] = []
    for f in fragments:
        if isinstance(f, SkillSpec):
            if skill_spec is not None:
                raise ValueError("duplicate skill fragment: only one .given(skill(...)) per case")
            skill_spec = f
        elif isinstance(f, LLMSpec):
            if llm_spec is not None:
                raise ValueError("duplicate llm fragment: only one .given(llm(...)) per case")
            llm_spec = f
        elif isinstance(f, ToolsSpec):
            tool_items.extend(f.items)
        else:
            raise TypeError(f"unknown agent fragment: {type(f).__name__}")

    if llm_spec is None:
        raise ValueError("missing llm fragment: attach an LLMClient via .given(llm(...))")

    return SmeltAgent(
        skill=skill_spec.source if skill_spec else None,
        system_prompt=skill_spec.prompt if skill_spec else None,
        llm=llm_spec.client,
        tools=tuple(tool_items),
        max_turns=max_turns,
    )
