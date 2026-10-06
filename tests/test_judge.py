"""Tests for the LLM-as-judge expectation."""

import pytest

from smelt import LLMResponse, ScriptedLLM, fixed_agent, llm_judge, new_case, text
from smelt.then.judge import JUDGE_PROMPT, LLMJudgeExpectation, _render_trace_calls
from smelt.trace import ToolCallRecord, Trace


def _judge_with(content: str) -> ScriptedLLM:
    return ScriptedLLM([LLMResponse.say(content)])


def _trace(output: str = "answer", calls=()) -> Trace:
    return Trace(output=output, tool_calls=[
        ToolCallRecord(name=c[0], arguments=c[1], error=c[2] if len(c) > 2 else None) for c in calls
    ])


def test_judge_requires_criteria_or_reference():
    with pytest.raises(ValueError, match="at least one"):
        LLMJudgeExpectation(judge=_judge_with("{}"))


def test_judge_pass_and_fail_by_threshold():
    judge = _judge_with('{"score": 0.9, "reason": "fully meets the criteria"}')
    ok = llm_judge(judge, criteria="the answer must be polite", threshold=0.8).evaluate(_trace())
    assert ok.passed and ok.score == 0.9 and "meets the criteria" in ok.message

    judge2 = _judge_with('{"score": 0.4, "reason": "not good enough"}')
    bad = llm_judge(judge2, criteria="the answer must be polite", threshold=0.8).evaluate(_trace())
    assert not bad.passed and bad.score == 0.4


def test_judge_receives_reference_and_criteria():
    judge = _judge_with('{"score": 1.0, "reason": "ok"}')
    llm_judge(judge, criteria="compare item by item", reference="the reference answer").evaluate(_trace("actual output"))
    prompt = judge.calls[0][0]["content"]
    assert "Criteria: compare item by item" in prompt
    assert "the reference answer" in prompt
    assert "actual output" in prompt


def test_judge_include_trace_sends_tool_calls():
    judge = _judge_with('{"score": 1.0}')
    trace = _trace("out", calls=[("run_command", {"cmd": "ls"}, None), ("bad_tool", {}, "boom")])
    llm_judge(judge, criteria="process is sound", include_trace=True).evaluate(trace)
    prompt = judge.calls[0][0]["content"]
    assert "run_command" in prompt and "bad_tool" in prompt and "boom" in prompt


def test_judge_unparseable_output_scores_zero():
    judge = _judge_with("looks fine to me")
    r = llm_judge(judge, criteria="x").evaluate(_trace())
    assert not r.passed and r.score == 0.0 and "not parseable" in r.message


def test_judge_score_out_of_range_scores_zero():
    judge = _judge_with('{"score": 1.5, "reason": "too excited"}')
    r = llm_judge(judge, criteria="x").evaluate(_trace())
    assert not r.passed and "out of range" in r.message


def test_judge_call_failure_scores_zero():
    class BoomLLM:
        def complete(self, messages, tools):
            raise RuntimeError("api down")

    r = llm_judge(BoomLLM(), criteria="x").evaluate(_trace())
    assert not r.passed and "judge call failed" in r.message and "api down" in r.message


def test_judge_in_full_case_pipeline():
    judge = _judge_with('{"score": 0.95, "reason": "accurate"}')
    result = (
        new_case("judge-case")
        .given(fixed_agent("Paris is the capital of France"))
        .when(text("what is the capital of France?"))
        .then(llm_judge(judge, reference="The capital of France is Paris", threshold=0.8))
        .run()
    )
    assert result.passed, result.summary()


def test_judge_receives_task_input_by_default():
    judge = _judge_with('{"score": 1.0, "reason": "ok"}')
    trace = Trace(output="Paris", messages=[
        {"role": "system", "content": "you are a geography skill"},
        {"role": "user", "content": "what is the capital of France?"},
    ])
    llm_judge(judge, criteria="answer must be correct").evaluate(trace)
    prompt = judge.calls[0][0]["content"]
    assert "what is the capital of France?" in prompt
    assert "hidden from the judge" not in prompt


def test_judge_include_input_false_hides_input():
    judge = _judge_with('{"score": 1.0}')
    trace = Trace(output="Paris", messages=[{"role": "user", "content": "secret question"}])
    llm_judge(judge, criteria="x", include_input=False).evaluate(trace)
    prompt = judge.calls[0][0]["content"]
    assert "secret question" not in prompt
    assert "hidden from the judge" in prompt


def test_judge_without_recorded_input_degrades_gracefully():
    judge = _judge_with('{"score": 1.0}')
    llm_judge(judge, criteria="x").evaluate(Trace(output="answer"))
    prompt = judge.calls[0][0]["content"]
    assert "no task input recorded" in prompt


def test_render_trace_calls_empty_and_error():
    assert _render_trace_calls(Trace()) == "(no tool calls)"
    trace = Trace(tool_calls=[ToolCallRecord(name="t", arguments={"a": 1}, error="boom")])
    assert "error: boom" in _render_trace_calls(trace)


def test_render_trace_calls_includes_tool_results():
    # groundedness: the judge must see what the tool actually returned
    trace = Trace(tool_calls=[
        ToolCallRecord(name="get_balance", arguments={"user": "alice"}, result={"balance_cents": 12400}),
    ])
    rendered = _render_trace_calls(trace)
    assert "12400" in rendered


def test_render_trace_calls_truncates_long_results():
    trace = Trace(tool_calls=[ToolCallRecord(name="read_file", arguments={}, result="x" * 5000)])
    rendered = _render_trace_calls(trace)
    assert len(rendered) < 1000
    assert "truncated" in rendered


def test_judge_prompt_template_is_valid_format_string():
    rendered = JUDGE_PROMPT.format(input_block="i", output_block="o", criteria_block="c")
    assert '"reason"' in rendered and '"score"' in rendered and "c" in rendered


# ---------------------------------------------------------------------------
# Multi-dimensional judging: one judge call per dimension, categorical scale
# ---------------------------------------------------------------------------


def test_judge_dimensions_scored_in_separate_calls():
    judge = ScriptedLLM([
        '{"reason": "tool result used correctly", "score": 2}',
        '{"reason": "no recovery after the error", "score": 0}',
    ])
    result = llm_judge(
        judge,
        dimensions=["result utilization (did the agent use tool output)", "error recovery"],
        threshold=0.4,
    ).evaluate(_trace("answer"))
    # each dimension is judged in its own call (no anchor bleed between dimensions)
    assert len(judge.calls) == 2
    assert "result utilization" in judge.calls[0][0]["content"]
    assert "error recovery" in judge.calls[1][0]["content"]
    assert "result utilization" not in judge.calls[1][0]["content"]
    # categorical 0/1/2 mapped to 0.0/0.5/1.0, then averaged
    assert result.score == pytest.approx(0.5)
    assert result.passed
    dims = result.details["dimensions"]
    assert [(d["name"], d["score"]) for d in dims] == [
        ("result utilization (did the agent use tool output)", 1.0),
        ("error recovery", 0.0),
    ]
    assert dims[0]["reason"] == "tool result used correctly"


def test_judge_dimensions_perfect_score_passes_high_threshold():
    judge = ScriptedLLM(['{"reason": "ok", "score": 2}', '{"reason": "ok", "score": 2}'])
    result = llm_judge(judge, dimensions=["a", "b"], threshold=0.9).evaluate(_trace())
    assert result.score == 1.0 and result.passed


def test_judge_dimension_invalid_level_scores_zero():
    judge = ScriptedLLM(['{"reason": "confused", "score": 5}', '{"reason": "ok", "score": 1}'])
    result = llm_judge(judge, dimensions=["a", "b"], threshold=0.9).evaluate(_trace())
    assert result.details["dimensions"][0]["score"] == 0.0
    assert "0|1|2" in result.details["dimensions"][0]["reason"] or "invalid" in result.details["dimensions"][0]["reason"]
    assert result.score == pytest.approx(0.25)  # (0.0 + 0.5) / 2


def test_judge_dimension_call_failure_scores_zero():
    class BoomLLM:
        def complete(self, messages, tools):
            raise RuntimeError("api down")

    result = llm_judge(BoomLLM(), dimensions=["a", "b"]).evaluate(_trace())
    assert result.score == 0.0
    assert all("api down" in d["reason"] for d in result.details["dimensions"])


def test_judge_requires_criteria_reference_or_dimensions():
    with pytest.raises(ValueError, match="at least one"):
        LLMJudgeExpectation(judge=_judge_with("{}"))


def test_judge_prompt_reason_before_score_and_length_neutral():
    prompt = JUDGE_PROMPT.format(input_block="i", output_block="o", criteria_block="c")
    # chain-of-thought: justification before the numeric verdict
    assert prompt.index('"reason"') < prompt.index('"score"')
    # verbosity bias: conciseness must not be penalized
    assert "concise" in prompt
