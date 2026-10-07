"""Reference testing demo: reach, restraint, ablation validity.

Usage: uv run smelt run examples/cases/reference_cases.py -v
All three cases pass (exit 0).
"""

import tempfile
from pathlib import Path

from smelt import (
    LLMResponse,
    ScriptedLLM,
    new_case,
    no_reference_read,
    output_contains,
    reference_folder,
    reference_read,
    reference_untouched,
    smelt_agent,
    text,
    tool,
)

# a demo skill whose depth lives in references/
SKILL_DIR = Path(tempfile.mkdtemp(prefix="smelt-ref-demo-"))
(SKILL_DIR / "references").mkdir()
(SKILL_DIR / "SKILL.md").write_text(
    """---
name: reddit
description: Query Reddit endpoints
---

# Reddit Skill

For exact parameter lists, read references/endpoints.md first — do not guess.
""",
    encoding="utf-8",
)
(SKILL_DIR / "references" / "endpoints.md").write_text(
    "# Endpoints\n- GET /r/{sub}/new — params: limit (1-100), after cursor\n",
    encoding="utf-8",
)


@tool
def read_file(path: str) -> str:
    """Read a file"""
    return Path(path).read_text(encoding="utf-8")


# 1) reach: the task needs exact params → the agent must consult the reference
reach = (
    new_case("ref-reach")
    .given(reference_folder(SKILL_DIR))
    .given(smelt_agent(
        SKILL_DIR,
        llm=ScriptedLLM([
            LLMResponse.call("read_file", {"path": "references/endpoints.md"}),
            LLMResponse.say("limit (1-100) and after cursor"),
        ]),
        tools=[read_file],
    ))
    .when(text("exact params of /new?"))
    .then(reference_read("references/endpoints.md"))
    .then(output_contains("limit"))
)

# 2) restraint: common question → the reference stays unread
restraint = (
    new_case("ref-restraint")
    .given(reference_folder(SKILL_DIR))
    .given(smelt_agent(
        SKILL_DIR,
        llm=ScriptedLLM([LLMResponse.say("/hot lists current hot posts")]),
        tools=[read_file],
    ))
    .when(text("what does /hot do?"))
    .then(no_reference_read("references/endpoints.md"))
)

# 3) ablation validity (without arm): nothing mounted, and the fingerprint
#    proves the content never entered the context
without = (
    new_case("ref-ablation-without")
    .given(smelt_agent(
        llm=ScriptedLLM([LLMResponse.say("I do not have the parameter list.")]),
        system_prompt="You answer Reddit API questions.",
        tools=[read_file],
    ))
    .when(text("exact params of /new?"))
    .then(reference_untouched("references/endpoints.md", source=SKILL_DIR))
)

cases = [reach, restraint, without]
