"""LLM-as-judge: let another model grade the output against a rubric.

The judge can be any LLMClient — ScriptedLLM (deterministic unit tests),
OpenAIChatClient (real models), LangChainLLM (langchain ecosystem) — sharing
the same seam as SmeltAgent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from smelt.given.agents.llm import LLMClient
from smelt.results import ExpectationResult
from smelt.then.expectations import _extract_json, _result
from smelt.trace import Trace

JUDGE_PROMPT = """You are a strict reviewer. Grade the "Answer under review" against the grading criteria.

## Task input
{input_block}

## Answer under review
{output_block}

## Grading criteria
{criteria_block}

Output JSON only, nothing else: {{"score": <a decimal between 0.0 and 1.0>, "reason": "<one-sentence justification>"}}"""


def _render_trace_calls(trace: Trace) -> str:
    lines = [
        f"- {c.name}({json.dumps(c.arguments, ensure_ascii=False)})"
        + (f" -> error: {c.error}" if c.error else "")
        for c in trace.tool_calls
    ]
    return "\n".join(lines) or "(no tool calls)"


@dataclass(frozen=True)
class LLMJudgeExpectation:
    """Let a judge model grade the agent output against criteria / a reference.

    - ``judge``: the judging LLMClient; may differ from the agent's model
      (e.g. a stronger model as judge);
    - ``criteria``: grading rubric, free text;
    - ``reference``: reference answer for the judge to compare against;
    - ``threshold``: judge score >= threshold passes;
    - ``include_trace``: also hand the tool-call trace to the judge
      (grade the process, not just the outcome).

    At least one of criteria / reference is required.
    """

    judge: LLMClient
    criteria: str | None = None
    reference: str | None = None
    threshold: float = 0.8
    include_trace: bool = False

    def __post_init__(self) -> None:
        if self.criteria is None and self.reference is None:
            raise ValueError("llm_judge requires at least one of criteria or reference")

    @property
    def name(self) -> str:
        what = self.criteria[:30] if self.criteria else f"reference={self.reference[:30]!r}"
        return f"llm_judge({what}, threshold={self.threshold})"

    def evaluate(self, trace: Trace) -> ExpectationResult:
        prompt = JUDGE_PROMPT.format(
            input_block="(hidden from the judge)",
            output_block=self._output_block(trace),
            criteria_block=self._criteria_block(),
        )
        try:
            response = self.judge.complete([{"role": "user", "content": prompt}], [])
        except Exception as e:  # noqa: BLE001 - a judge failure must not crash the whole case
            return _result(self.name, 0.0, self.threshold, f"judge call failed: {type(e).__name__}: {e}")
        try:
            parsed = _extract_json(response.content)
            score = float(parsed["score"])
            reason = str(parsed.get("reason", ""))
        except (ValueError, KeyError, TypeError) as e:
            return _result(
                self.name, 0.0, self.threshold,
                f"judge output is not parseable as {{score, reason}}: {e}; raw: {response.content[:200]!r}",
            )
        if not 0.0 <= score <= 1.0:
            return _result(self.name, 0.0, self.threshold, f"judge score out of range: {score}")
        return _result(self.name, score, self.threshold, reason, judge_score=score, reason=reason)

    def _output_block(self, trace: Trace) -> str:
        if not self.include_trace:
            return trace.output
        return f"Final output:\n{trace.output}\n\nTool-call trace:\n{_render_trace_calls(trace)}"

    def _criteria_block(self) -> str:
        parts = []
        if self.criteria:
            parts.append(f"Criteria: {self.criteria}")
        if self.reference:
            parts.append(f"Reference answer:\n{self.reference}")
        return "\n\n".join(parts)


def llm_judge(
    judge: LLMClient,
    *,
    criteria: str | None = None,
    reference: str | None = None,
    threshold: float | None = None,
    include_trace: bool = False,
) -> LLMJudgeExpectation:
    """then(llm_judge(judge_llm, criteria="must politely decline")).

    When threshold is omitted, the global default
    ``smelt.config.llm_judge_threshold`` applies (initially 0.8; change via
    ``smelt.configure(llm_judge_threshold=...)``).
    """
    from smelt.config import config

    return LLMJudgeExpectation(
        judge=judge,
        criteria=criteria,
        reference=reference,
        threshold=threshold if threshold is not None else config.llm_judge_threshold,
        include_trace=include_trace,
    )
