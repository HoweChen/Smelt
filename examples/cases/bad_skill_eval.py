"""Bad-skill evaluation demo: a bad skill with bad references, scored end to end.

Runs offline (ScriptedLLM everywhere). Four behavior cases exhibit four failure
modes; lint catches the dangling reference and the TODO marker; the judge scores
the writing poorly. The report is saved to examples/reports/bad_ref_skill_eval.md
and self-checked at the end (deterministic — the asserts are stable).

    uv run python examples/cases/bad_skill_eval.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from smelt import (
    LLMResponse,
    ScriptedLLM,
    evaluate_skill,
    new_case,
    no_reference_read,
    reference_folder,
    reference_read,
    reference_untouched,
    smelt_agent,
    text,
    tool,
    tool_budget,
)

SKILL = Path(__file__).resolve().parents[1] / "bad_ref_skill"


@tool
def read_file(path: str) -> str:
    """Read a file"""
    return Path(path).read_text(encoding="utf-8")


# --- behavior cases: four failure modes, all deliberate ---------------------

# 1) reach FAIL: the task needs exact params, but the agent answers from memory
reach_fail = (
    new_case("reach-fail")
    .given(reference_folder(SKILL))
    .given(smelt_agent(SKILL, llm=ScriptedLLM([LLMResponse.say("just pass ?limit=50")]), tools=[read_file]))
    .when(text("exact params of the /new endpoint?"))
    .then(reference_read("references/endpoints.md"))
)

# 2) restraint FAIL: the (bad) skill says "always read for every question" —
#    so the agent reads the reference even for a trivial question
restraint_fail = (
    new_case("restraint-fail")
    .given(reference_folder(SKILL))
    .given(smelt_agent(SKILL, llm=ScriptedLLM([
        LLMResponse.call("read_file", {"path": "references/endpoints.md"}),
        LLMResponse.say("/hot lists current hot posts"),
    ]), tools=[read_file]))
    .when(text("what does /hot do?"))
    .then(no_reference_read("references/endpoints.md"))
)

# 3) ablation FAIL: nothing is mounted, yet the agent recites the reference
#    content verbatim — parametric-memory contamination, arm invalid
ablation_fail = (
    new_case("ablation-contaminated")
    .given(smelt_agent(
        llm=ScriptedLLM([LLMResponse.say("GET /r/{sub}/hot — no params needed")]),
        system_prompt="You answer Reddit API questions.",
        tools=[read_file],
    ))
    .when(text("exact params of the /hot endpoint?"))
    .then(reference_untouched("references/endpoints.md", source=SKILL))
)

# 4) budget FAIL: the agent loops, re-reading the same file three times
budget_fail = (
    new_case("budget-fail")
    .given(reference_folder(SKILL))
    .given(smelt_agent(SKILL, llm=ScriptedLLM([
        LLMResponse.call("read_file", {"path": "references/endpoints.md"}),
        LLMResponse.call("read_file", {"path": "references/endpoints.md"}),
        LLMResponse.call("read_file", {"path": "references/endpoints.md"}),
        LLMResponse.say("done"),
    ]), tools=[read_file]))
    .when(text("summarize the endpoints"))
    .then(tool_budget("read_file", max=1))
)

# --- judge: scores the (bad) writing poorly ---------------------------------

judge = ScriptedLLM([
    (
        '{"dimensions": ['
        '{"name": "Metadata & naming", "score": 0.2, "comment": "description \\"does stuff\\" is meaningless"},'
        '{"name": "Trigger guidance", "score": 0.1, "comment": "no trigger scenarios; a TODO marker remains"},'
        '{"name": "Structure & readability", "score": 0.4, "comment": "minimal structure, dangling link"},'
        '{"name": "Examples & edge cases", "score": 0.0, "comment": "none at all"},'
        '{"name": "Actionability", "score": 0.3, "comment": "over-reads references; points at a missing file"}],'
        '"overall_comment": "rewrite the description and fix the dangling reference"}'
    ),
    (
        '{"suggestions": ['
        '"replace the description with concrete trigger scenarios", '
        '"create references/missing.md or remove the pointer", '
        '"drop the always-read instruction so references load on demand"]}'
    ),
])

report = (
    evaluate_skill(SKILL, judge=judge)
    .with_cases(reach_fail, restraint_fail, ablation_fail, budget_fail)
    .with_times(1)  # scripted agents are deterministic anyway; keep it explicit
    .run()
)

out = report.save(Path(__file__).resolve().parents[1] / "reports" / "bad_ref_skill_eval.md")

print(f"behavior: {report.behavior_score * 100:.1f}")
print(f"writing:  {report.writing_score * 100:.1f}")
print(f"lint:     {report.lint_score * 100:.1f}")
print(f"overall:  {report.overall_score:.1f} ({report.grade})")
print(f"report:   {out}")

# --- self-check: the report must reflect the four planted failures -----------

md = report.to_markdown()
assert report.grade == "F", report.grade
assert all(not r.passed for r in report.behavior_results), "all four cases should fail"
assert "contamination" in md  # the ablation arm is flagged as contaminated
assert "## Reference Coverage" in md and "unreached" in md
coverage = {c["path"]: c for c in report.reference_coverage}
assert coverage["references/endpoints.md"]["reached"] >= 1
assert coverage["references/missing.md"]["origin"] == "scanned"  # dangling pointer surfaced by the scan
assert "references/missing.md" in md  # lint reports the dangling asset
print("\nself-check passed: all four planted failures are visible in the report")
