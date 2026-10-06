"""Smelt — a behavior-verification framework for agent skills.

Put a skill in the furnace: given contexts and an agent, when a trigger fires,
then assert on the outcome — one consistent yardstick to verify that each skill
version behaves as intended.

Minimal example::

    from smelt import new_case, smelt_agent, text, tool_call

    def test_commit_skill():
        result = (
            new_case("commit")
            .given(smelt_agent("skills/commit", llm=my_llm, tools=[git_tool]))
            .when(text("commit my changes"))
            .then(tool_call("run_command"))
            .run()
        )
        result.assert_passed()
"""

__version__ = "0.2.0"

from smelt.case import SmeltCase, new_case
from smelt.config import SmeltConfig, config, configure
from smelt.evaluate import (
    SkillEvaluation,
    SkillEvaluationBuilder,
    WritingAssessment,
    WritingDimension,
    evaluate_skill,
)
from smelt.given import (
    Agent,
    CaseContext,
    ContextSpec,
    FixedAgent,
    LLMClient,
    LLMResponse,
    LLMSpec,
    OpenAIChatClient,
    ScriptedLLM,
    SkillSpec,
    SmeltAgent,
    ToolCall,
    ToolsSpec,
    context,
    fixed_agent,
    llm,
    skill,
    smelt_agent,
    tools,
)
from smelt.results import CaseResult, ExpectationResult
from smelt.runner import run_case
from smelt.then import (
    Expectation,
    LLMJudgeExpectation,
    json_output,
    llm_judge,
    no_tool_call,
    output_contains,
    output_equals,
    text_similar,
    tool_call,
)
from smelt.tools import Tool, tool
from smelt.trace import ToolCallRecord, Trace
from smelt.when import CaseInput, DirectoryInput, TextInput, directory, text

__all__ = [
    "Agent",
    "CaseContext",
    "CaseInput",
    "CaseResult",
    "ContextSpec",
    "DirectoryInput",
    "Expectation",
    "ExpectationResult",
    "FixedAgent",
    "LLMClient",
    "LLMJudgeExpectation",
    "LLMResponse",
    "LLMSpec",
    "OpenAIChatClient",
    "ScriptedLLM",
    "SkillEvaluation",
    "SkillEvaluationBuilder",
    "SkillSpec",
    "SmeltAgent",
    "SmeltCase",
    "SmeltConfig",
    "TextInput",
    "Tool",
    "ToolCall",
    "ToolCallRecord",
    "ToolsSpec",
    "Trace",
    "WritingAssessment",
    "WritingDimension",
    "__version__",
    "config",
    "configure",
    "context",
    "directory",
    "evaluate_skill",
    "fixed_agent",
    "json_output",
    "llm",
    "llm_judge",
    "new_case",
    "no_tool_call",
    "output_contains",
    "output_equals",
    "run_case",
    "skill",
    "smelt_agent",
    "text",
    "text_similar",
    "tool",
    "tool_call",
    "tools",
]
