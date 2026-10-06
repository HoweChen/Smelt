"""Runner: materialize context → execute agent → evaluate expectations."""

from __future__ import annotations

import re
from pathlib import Path

from smelt.case import SmeltCase
from smelt.given.context import CaseContext, fresh_workspace
from smelt.results import CaseResult


def _slug(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", name).strip("-") or "case"


def run_case(case: SmeltCase) -> CaseResult:
    """Execute one case. Any configuration/runtime error converges into CaseResult.error."""
    agent = case.agent
    if agent is None and case.fragments:
        from smelt.given.fragments import assemble_agent

        try:
            agent = assemble_agent(case.fragments)
        except (ValueError, TypeError) as e:
            return CaseResult(case_name=case.name, expectations=[], error=str(e))
    if agent is None:
        return CaseResult(
            case_name=case.name,
            expectations=[],
            error="missing agent: attach one via .given(smelt_agent(...) / fixed_agent(...)) "
            "or fragments .given(skill() / llm() / tools())",
        )
    if case.trigger is None:
        return CaseResult(
            case_name=case.name,
            expectations=[],
            error="missing trigger: set one via .when(text(...) | directory(...))",
        )

    workspace = fresh_workspace(
        Path(".smelt") / _slug(case.name) if case.keep_workspace else None,
        name=_slug(case.name),
    )
    ctx = CaseContext(workspace=workspace)
    try:
        ctx.materialize(list(case.contexts))
        trace = agent.run(ctx, case.trigger)
    except Exception as e:  # noqa: BLE001 - the framework stays stable; errors land in the result
        return CaseResult(
            case_name=case.name,
            expectations=[],
            error=f"{type(e).__name__}: {e}",
            workspace=str(workspace),
        )

    results = [e.evaluate(trace) for e in case.expectations]
    return CaseResult(
        case_name=case.name,
        expectations=results,
        trace=trace,
        workspace=str(workspace),
    )
