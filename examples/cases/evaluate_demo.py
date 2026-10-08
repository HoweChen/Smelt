"""evaluate_skill comprehensive-review demo (runs offline; the judge is a ScriptedLLM).

For real use, swap judge / agent_llm for OpenAIChatClient or LangChainLLM:

    uv run python examples/cases/evaluate_demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from smelt import ScriptedLLM, evaluate_skill, fixed_agent, new_case, output_contains, text, tool_call

GOOD_SKILL = Path(__file__).resolve().parents[1] / "good-skill"

# Real world: judge = OpenAIChatClient("kimi-k2", base_url="https://api.moonshot.cn/v1")
judge = ScriptedLLM([
    # writing-review response
    (
        '{"dimensions": ['
        '{"name": "Semantic accuracy", "score": 0.95, "comment": "name and description capture the purpose well"},'
        '{"name": "Examples & edge cases", "score": 0.6, "comment": "missing a failure-path example"},'
        '{"name": "Actionability", "score": 0.8, "comment": "steps are explicit but some commands lack parameter notes"}],'
        '"overall_comment": "good quality; add failure-path examples"}'
    ),
    # suggestion-generation response
    (
        '{"suggestions": ["add a failure-path invocation example under examples/", '
        '"document command parameters in a table", '
        '"split the caveats section into a dedicated edge-cases chapter"]}'
    ),
])

report = (
    evaluate_skill(GOOD_SKILL, judge=judge)
    .with_cases(
        # behavior case: replaying a known-good trace (real world: smelt_agent + a real llm)
        new_case("hello-script")
        .given(fixed_agent(
            "hello world",
            tool_calls=[{"name": "run_command", "arguments": {"cmd": "python scripts/hello.py"}}],
        ))
        .when(text("run the hello script"))
        .then(tool_call("run_command"))
        .then(output_contains("hello")),
    )
    .with_suggestions(max_items=3)
    .run()
)

out = report.save(Path(__file__).resolve().parents[1] / "reports" / "good-skill_eval.md")
print(f"overall {report.overall_score:.1f} ({report.grade})")
print(f"report written to {out}")
