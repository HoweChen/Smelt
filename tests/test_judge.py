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


def test_render_trace_calls_empty_and_error():
    assert _render_trace_calls(Trace()) == "(no tool calls)"
    trace = Trace(tool_calls=[ToolCallRecord(name="t", arguments={"a": 1}, error="boom")])
    assert "error: boom" in _render_trace_calls(trace)


def test_judge_prompt_template_is_valid_format_string():
    rendered = JUDGE_PROMPT.format(input_block="i", output_block="o", criteria_block="c")
    assert '{"score"' in rendered and "c" in rendered
