"""Runner: materialize context → execute agent → evaluate expectations.

Repeated sampling: ``run_case(case, times=N)`` executes the case N times and
aggregates — case score = mean of per-run scores (a crashed run counts as 0),
each expectation's score = mean across the runs that produced it. The spread
(std) is what tells you whether a score is stable enough to compare versions.
"""

from __future__ import annotations

import re
import time
from dataclasses import replace
from pathlib import Path

from smelt.case import SmeltCase
from smelt.given.context import CaseContext, fresh_workspace
from smelt.results import CaseResult, ExpectationResult


def _slug(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", name).strip("-") or "case"


def run_case(case: SmeltCase, times: int | None = None) -> CaseResult:
    """Execute one case, ``times`` times (default: case.times, initially 1).

    Configuration errors (missing agent/trigger, bad fragments) converge into
    CaseResult.error without repeating — there is nothing to sample.
    """
    n = times if times is not None else case.times
    if n < 1:
        raise ValueError(f"times must be >= 1, got {n}")
    first = _run_once(case)
    if n == 1 or (first.error is not None and first.workspace is None):
        return first
    return _aggregate(case.name, [first, *(_run_once(case) for _ in range(n - 1))])


def _run_once(case: SmeltCase) -> CaseResult:
    """Execute one case once. Any configuration/runtime error converges into CaseResult.error."""
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
        start = time.monotonic()
        trace = agent.run(ctx, case.trigger)
        trace.wall_time_s = time.monotonic() - start
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
        references=list(ctx.references),
    )


def _aggregate(name: str, results: list[CaseResult]) -> CaseResult:
    """Aggregate N runs: case score = mean of run scores (errors count as 0);
    expectation scores average over the runs that evaluated them;
    run_passed feeds the pass^k reliability estimate."""
    run_scores = [r.score for r in results]
    run_passed = [r.passed for r in results]
    run_errors = [r.error for r in results if r.error is not None]
    ok = [r for r in results if r.error is None]

    if not ok:
        return CaseResult(
            case_name=name,
            expectations=[],
            error=f"all {len(results)} runs failed; first error: {run_errors[0]}",
            workspace=results[-1].workspace,
            runs=len(results),
            run_scores=run_scores,
            run_errors=run_errors,
            run_passed=run_passed,
        )

    aggregated: list[ExpectationResult] = []
    for i in range(len(ok[0].expectations)):
        parts = [r.expectations[i] for r in ok]
        scores = tuple(p.score for p in parts)
        message = next((p.message for p in parts if p.message), "")
        aggregated.append(replace(
            parts[0],
            score=sum(scores) / len(scores),
            message=message,
            runs=len(parts),
            run_scores=scores,
        ))

    last = ok[-1]
    merged_trace = None
    if last.trace is not None:
        # merge every successful run's tool calls so coverage/detection sees all runs
        merged_trace = replace(
            last.trace,
            tool_calls=[c for r in ok for c in (r.trace.tool_calls if r.trace else [])],
        )
    return CaseResult(
        case_name=name,
        expectations=aggregated,
        trace=merged_trace,
        workspace=last.workspace,
        runs=len(results),
        run_scores=run_scores,
        run_errors=run_errors,
        run_passed=run_passed,
        references=last.references,
    )
