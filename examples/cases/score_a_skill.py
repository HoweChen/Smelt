"""Minimal end-to-end example: verify a skill and get its score (with tool calling).

Run: uv run python examples/cases/score_a_skill.py

Two paths demonstrated:
1. Deterministic replay (ScriptedLLM) — the CI default, no network;
2. Real models — swap ScriptedLLM for OpenAIChatClient / LangChainLLM, nothing else changes.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from smelt import (
    LLMResponse,
    ScriptedLLM,
    evaluate_skill,
    llm_judge,
    new_case,
    output_contains,
    smelt_agent,
    text,
    text_similar,
    tool,
    tool_call,
)

# ---------------------------------------------------------------------------
# 1. Define the tools available to the agent (type hints generate the JSON
#    Schema automatically; handlers run inside an isolated workspace)
# ---------------------------------------------------------------------------


@tool
def read_file(path: str) -> str:
    """Read a file from the workspace"""
    return Path(path).read_text(encoding="utf-8")


@tool
def write_file(path: str, content: str) -> str:
    """Write a file into the workspace"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"wrote {p}"


# ---------------------------------------------------------------------------
# 2. Prepare the skill under test (a "meeting notes" skill created on the fly)
# ---------------------------------------------------------------------------

skill_dir = Path(__file__).resolve().parent / "_demo_skill"
skill_dir.mkdir(exist_ok=True)
(skill_dir / "SKILL.md").write_text(
    """---
name: meeting-notes
description: Use this skill when the user provides messy meeting records; distill them into structured minutes (decisions, action items, owners) written to notes.md
---

# Meeting Notes

## When to use

When the user asks to organize meeting records or distill minutes, use this skill.

## Steps

1. Use read_file to read the raw record file the user points to;
2. Distill "Decisions" and "Action items (with owners)";
3. Use write_file to write the result to notes.md, then report the highlights.
""",
    encoding="utf-8",
)

# ---------------------------------------------------------------------------
# 3. LLMs: ScriptedLLM replays in tests (swap for OpenAIChatClient in real
#    regression runs)
# ---------------------------------------------------------------------------

# agent_llm = OpenAIChatClient(model="kimi-k2", base_url="https://api.moonshot.cn/v1")
agent_llm = ScriptedLLM([
    LLMResponse.call("read_file", {"path": "raw.txt"}),
    LLMResponse.call("write_file", {
        "path": "notes.md",
        "content": "# Minutes\n## Decisions\n- Ship v2 next Friday\n## Action items\n- Alice: own the rollout plan\n",
    }),
    LLMResponse.say("Done — notes.md written: 1 decision (ship v2 next Friday), 1 action item (Alice owns the rollout plan)."),
])

# The judge can also be a real model; replay order = consumption order:
# ① llm_judge inside the behavior case → ② writing review → ③ suggestions
judge_llm = ScriptedLLM([
    '{"score": 0.9, "reason": "highlights complete and accurate"}',
    (
        '{"dimensions": ['
        '{"name": "Metadata & naming", "score": 0.9, "comment": "accurate description"},'
        '{"name": "Trigger guidance", "score": 0.85, "comment": "clear trigger scenarios"},'
        '{"name": "Structure & readability", "score": 0.9, "comment": "well structured"},'
        '{"name": "Examples & edge cases", "score": 0.5, "comment": "no note on missing input files"},'
        '{"name": "Actionability", "score": 0.85, "comment": "steps are executable"}],'
        '"overall_comment": "good; add edge-case handling"}'
    ),
    '{"suggestions": ["document what to do when the raw file is missing", "give a fixed template example for notes.md"]}',
])

# ---------------------------------------------------------------------------
# 4. Behavior case: given context & agent → when trigger → then assertions
#    (tool calling)
# ---------------------------------------------------------------------------

fixture = Path(__file__).resolve().parent / "_demo_fixture"
fixture.mkdir(exist_ok=True)
(fixture / "raw.txt").write_text("Alice says v2 ships next Friday; she also owns the rollout plan.", encoding="utf-8")

behavior_case = (
    new_case("distill meeting notes")
    .given(smelt_agent(skill_dir, llm=agent_llm, tools=[read_file, write_file]))
    .when(text("distill raw.txt into meeting minutes"))  # or when(directory("some/dir"))
    .then(tool_call("read_file", args={"path": "raw.txt"}))          # assert the read
    .then(tool_call("write_file"))                                    # assert the write
    .then(output_contains("decision", "action item"))                 # key sections present
    .then(text_similar("finished; contains decisions and action items", threshold=0.4))  # similarity
    .then(llm_judge(judge_llm, criteria="the report must cover decisions and action items", threshold=0.7))  # LLM judge
)

# ---------------------------------------------------------------------------
# 5. Comprehensive evaluation: behavior + writing + lint → overall score & report
# ---------------------------------------------------------------------------

report = (
    evaluate_skill(skill_dir, judge=judge_llm)
    .with_cases(behavior_case)
    .run()
)

print(f"behavior: {report.behavior_score * 100:.1f}")
print(f"writing:  {report.writing_score * 100:.1f}")
print(f"lint:     {report.lint_score * 100:.1f}")
print(f"overall:  {report.overall_score:.1f} ({report.grade})")
print("\nsuggestions:")
for i, s in enumerate(report.suggestions, 1):
    print(f"  {i}. {s}")

out = report.save(Path(__file__).resolve().parents[1] / "reports" / "meeting_notes_eval.md")
print(f"\nfull report: {out}")
