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
from smelt.given.references import reference, reference_folder

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
    "reference",
    "reference_folder",
    "skill",
    "smelt_agent",
    "tools",
]
