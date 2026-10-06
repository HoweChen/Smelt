"""given — the precondition side of a case: contexts, agents, agent fragments (skill / llm / tools)."""

from smelt.given.agents import (
    Agent,
    FixedAgent,
    LLMClient,
    LLMResponse,
    OpenAIChatClient,
    ScriptedLLM,
    SmeltAgent,
    ToolCall,
    fixed_agent,
    smelt_agent,
)
from smelt.given.context import CaseContext, ContextSpec, context
from smelt.given.fragments import LLMSpec, SkillSpec, ToolsSpec, llm, skill, tools

__all__ = [
    "Agent",
    "CaseContext",
    "ContextSpec",
    "FixedAgent",
    "LLMClient",
    "LLMResponse",
    "LLMSpec",
    "OpenAIChatClient",
    "ScriptedLLM",
    "SkillSpec",
    "SmeltAgent",
    "ToolCall",
    "ToolsSpec",
    "context",
    "fixed_agent",
    "llm",
    "skill",
    "smelt_agent",
    "tools",
]
