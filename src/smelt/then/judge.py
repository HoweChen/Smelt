"""LLM-as-judge: let another model grade the output against a rubric.

The judge can be any LLMClient — ScriptedLLM (deterministic unit tests),
OpenAIChatClient (real models), LangChainLLM (langchain ecosystem) — sharing
the same seam as SmeltAgent.

Design notes (evidence-backed):
- reason before score (chain-of-thought judging reduces arbitrary verdicts);
- length neutrality is stated explicitly (judge verbosity bias);
- ``dimensions=[...]`` scores each dimension in a SEPARATE judge call —
  scoring several dimensions in one prompt lets the first dimension's anchor
  bleed into the rest, and failures become unattributable;
- dimension scoring uses a categorical 0|1|2 scale (fail / partial / pass)
  instead of a free decimal — coarse scales agree with human judgment better
  than fine-grained floats;
- the trace rendering includes tool RESULTS, not just call names — without
  them the judge cannot tell whether the agent used the tool output or
  hallucinated from memory (groundedness).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
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

Rules:
- Base the verdict only on evidence in the answer (and tool calls, if shown); do not reward guesswork.
- Length neutrality: at equal correctness, a concise answer scores at least as high as a verbose one.

Reason first, then score. Output JSON only, nothing else: {{"reason": "<one-sentence justification citing evidence>", "score": <a decimal between 0.0 and 1.0>}}"""

DIMENSION_PROMPT = """You are a strict reviewer. Grade the "Answer under review" on ONE dimension only.

## Task input
{input_block}

## Answer under review
{output_block}

## Dimension under review
{dimension}
{shared_block}
## Grading scale (integers only)
0 = fail: the answer clearly violates the dimension
1 = partial: the answer is partially right or the evidence is inconclusive
2 = pass: the answer fully satisfies the dimension

Rules:
- Base the verdict only on evidence in the answer (and tool calls, if shown); do not reward guesswork.
- Length neutrality: at equal correctness, a concise answer scores at least as high as a verbose one.

Reason first, then score. Output JSON only, nothing else: {{"reason": "<one-sentence justification citing evidence>", "score": <0|1|2>}}"""

_LEVEL_TO_SCORE = {0: 0.0, 1: 0.5, 2: 1.0}
_MAX_RESULT_CHARS = 500


def _render_trace_calls(trace: Trace) -> str:
    """Render tool calls for the judge, including (truncated) tool results —
    groundedness grading is impossible without seeing what the tool returned."""
    lines = []
    for c in trace.tool_calls:
        line = f"- {c.name}({json.dumps(c.arguments, ensure_ascii=False)})"
        if c.error:
            line += f" -> error: {c.error}"
        elif c.result is not None:
            payload = json.dumps(c.result, ensure_ascii=False, default=str)
            if len(payload) > _MAX_RESULT_CHARS:
                payload = payload[:_MAX_RESULT_CHARS] + f"... (truncated, {len(payload)} chars total)"
            line += f" -> {payload}"
        lines.append(line)
    return "\n".join(lines) or "(no tool calls)"


def _render_input(trace: Trace) -> str:
    """Recover the task input (first user message) from the recorded conversation."""
    for message in trace.messages:
        if message.get("role") == "user":
            return str(message.get("content", ""))
    return "(no task input recorded)"


@dataclass(frozen=True)
class LLMJudgeExpectation:
    """Let a judge model grade the agent output against criteria / a reference / dimensions.

    - ``judge``: the judging LLMClient; may differ from the agent's model
      (e.g. a stronger model as judge — preferably a different model family,
      judges rate their own family's output higher). Keep the judge at
      temperature 0 (OpenAIChatClient's default) so scores stay comparable
      across runs;
    - ``criteria``: grading rubric, free text;
    - ``reference``: reference answer for the judge to compare against;
    - ``dimensions``: independent quality dimensions (e.g. "result
      utilization", "error recovery"); each is graded in a separate judge
      call on a categorical 0|1|2 scale, and the expectation score is the
      mean. Given together with criteria/reference, those are shared context
      for every dimension call;
    - ``threshold``: judge score >= threshold passes;
    - ``include_input``: hand the task input (the when-trigger) to the judge —
      grading an answer without seeing the question inflates scores and blurs
      version differences (default True);
    - ``include_trace``: also hand the tool-call trace (including tool
      results) to the judge (grade the process, not just the outcome).

    At least one of criteria / reference / dimensions is required.
    """

    judge: LLMClient
    criteria: str | None = None
    reference: str | None = None
    dimensions: tuple[str, ...] = ()
    threshold: float = 0.8
    include_input: bool = True
    include_trace: bool = False

    def __post_init__(self) -> None:
        if self.criteria is None and self.reference is None and not self.dimensions:
            raise ValueError("llm_judge requires at least one of criteria, reference, or dimensions")

    @property
    def name(self) -> str:
        if self.dimensions:
            return f"llm_judge({len(self.dimensions)} dimensions, threshold={self.threshold})"
        what = self.criteria[:30] if self.criteria else f"reference={self.reference[:30]!r}"
        return f"llm_judge({what}, threshold={self.threshold})"

    def evaluate(self, trace: Trace) -> ExpectationResult:
        if self.dimensions:
            return self._evaluate_dimensions(trace)
        prompt = JUDGE_PROMPT.format(
            input_block=_render_input(trace) if self.include_input else "(hidden from the judge)",
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

    def _evaluate_dimensions(self, trace: Trace) -> ExpectationResult:
        """One judge call per dimension on a categorical 0|1|2 scale; mean of
        mapped scores (0.0 / 0.5 / 1.0) is the expectation score."""
        graded = []
        for dimension in self.dimensions:
            graded.append(self._grade_dimension(dimension, trace))
        score = sum(d["score"] for d in graded) / len(graded)
        weakest = min(graded, key=lambda d: d["score"])
        message = f"weakest: {weakest['name']} ({weakest['score']:.1f}) — {weakest['reason']}"
        return _result(self.name, score, self.threshold, message, dimensions=graded)

    def _grade_dimension(self, dimension: str, trace: Trace) -> dict:
        prompt = DIMENSION_PROMPT.format(
            input_block=_render_input(trace) if self.include_input else "(hidden from the judge)",
            output_block=self._output_block(trace),
            dimension=dimension,
            shared_block=self._shared_block(),
        )
        try:
            response = self.judge.complete([{"role": "user", "content": prompt}], [])
            parsed = _extract_json(response.content)
            level = int(parsed["score"])
            reason = str(parsed.get("reason", ""))
        except Exception as e:  # noqa: BLE001 - one failed dimension must not crash the case
            return {"name": dimension, "score": 0.0, "reason": f"judge failed: {type(e).__name__}: {e}"}
        if level not in _LEVEL_TO_SCORE:
            return {"name": dimension, "score": 0.0, "reason": f"invalid level {level} (expected 0|1|2); raw reason: {reason}"}
        return {"name": dimension, "score": _LEVEL_TO_SCORE[level], "reason": reason}

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

    def _shared_block(self) -> str:
        shared = self._criteria_block()
        return f"\n## Shared grading context\n{shared}\n" if shared else ""


def llm_judge(
    judge: LLMClient,
    *,
    criteria: str | None = None,
    reference: str | None = None,
    dimensions: Sequence[str] | None = None,
    threshold: float | None = None,
    include_input: bool = True,
    include_trace: bool = False,
) -> LLMJudgeExpectation:
    """then(llm_judge(judge_llm, criteria="must politely decline")).

    Or per-dimension grading (one judge call each, categorical scale)::

        llm_judge(judge_llm, dimensions=["result utilization", "error recovery"])

    When threshold is omitted, the global default
    ``smelt.config.llm_judge_threshold`` applies (initially 0.8; change via
    ``smelt.configure(llm_judge_threshold=...)``).
    """
    from smelt.config import config

    return LLMJudgeExpectation(
        judge=judge,
        criteria=criteria,
        reference=reference,
        dimensions=tuple(dimensions or ()),
        threshold=threshold if threshold is not None else config.llm_judge_threshold,
        include_input=include_input,
        include_trace=include_trace,
    )
