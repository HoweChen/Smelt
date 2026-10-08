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

__version__ = "0.4.2"

from smelt.case import SmeltCase, new_case
from smelt.challenge.doctor import DoctorReport, doctor
from smelt.challenge.guards import mutate_check
from smelt.challenge.probes import ChallengeResult
from smelt.compare import CaseDiff, CompareResult, compare
from smelt.config import SmeltConfig, config, configure
from smelt.env import load_env
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
    reference,
    reference_folder,
    skill,
    smelt_agent,
    tools,
)
from smelt.llm_config import LLMConfig, LLMConfigError
from smelt.report.suite import Suite, SuiteResult, suite
from smelt.results import CaseResult, ExpectationResult
from smelt.runner import run_case
from smelt.then import (
    Expectation,
    LLMJudgeExpectation,
    json_output,
    llm_judge,
    no_reference_read,
    no_tool_call,
    output_contains,
    output_equals,
    reference_read,
    reference_untouched,
    text_similar,
    tool_budget,
    tool_call,
    turns_used,
    wall_time,
)
from smelt.tools import Tool, mock_tool, tool
from smelt.trace import ToolCallRecord, Trace
from smelt.when import CaseInput, DirectoryInput, TextInput, directory, text

__all__ = [
    "Agent",
    "CaseContext",
    "CaseDiff",
    "CaseInput",
    "CaseResult",
    "ChallengeResult",
    "CompareResult",
    "ContextSpec",
    "DirectoryInput",
    "DoctorReport",
    "Expectation",
    "ExpectationResult",
    "FixedAgent",
    "LLMClient",
    "LLMConfig",
    "LLMConfigError",
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
    "Suite",
    "SuiteResult",
    "TextInput",
    "Tool",
    "ToolCall",
    "ToolCallRecord",
    "ToolsSpec",
    "Trace",
    "WritingAssessment",
    "WritingDimension",
    "__version__",
    "compare",
    "config",
    "configure",
    "context",
    "directory",
    "doctor",
    "evaluate_skill",
    "fixed_agent",
    "json_output",
    "llm",
    "llm_judge",
    "load_env",
    "mock_tool",
    "mutate_check",
    "new_case",
    "no_reference_read",
    "no_tool_call",
    "output_contains",
    "output_equals",
    "reference",
    "reference_folder",
    "reference_read",
    "reference_untouched",
    "run_case",
    "skill",
    "smelt_agent",
    "suite",
    "text",
    "text_similar",
    "tool",
    "tool_budget",
    "tool_call",
    "tools",
    "turns_used",
    "wall_time",
]
