"""Tests for repeated sampling: run a case N times and aggregate mean/std.

Motivation: LLM agents and LLM judges are stochastic — a single run's score is
noise. Repeating a case N times and aggregating (mean ± std) is what makes
scores comparable across skill versions.
"""

from pathlib import Path

import pytest

from smelt import (
    LLMResponse,
    ScriptedLLM,
    evaluate_skill,
    fixed_agent,
    new_case,
    output_contains,
    output_equals,
    smelt_agent,
    text,
)
from smelt.trace import Trace

ROOT = Path(__file__).resolve().parent.parent
GOOD = ROOT / "examples" / "good-skill"


class CyclingAgent:
    """Deterministic-per-call agent: returns outputs[i % n] on run i."""

    def __init__(self, outputs):
        self._outputs = list(outputs)
        self.calls = 0

    def run(self, ctx, input):
        out = self._outputs[self.calls % len(self._outputs)]
        self.calls += 1
        return Trace(output=out)


class FlakyAgent:
    """Raises on the given 1-based call numbers."""

    def __init__(self, fail_on):
        self._fail_on = set(fail_on)
        self.calls = 0

    def run(self, ctx, input):
        self.calls += 1
        if self.calls in self._fail_on:
            raise RuntimeError("boom")
        return Trace(output="ok")


def _case(agent, *expectations, name="rep"):
    return new_case(name).given(agent).when(text("go")).then(*expectations)


# ---------------------------------------------------------------------------
# Aggregation semantics
# ---------------------------------------------------------------------------


def test_repeat_averages_case_scores_across_runs():
    case = _case(CyclingAgent(["ab", "a", ""]), output_contains("a", "b"))
    result = case.run(times=3)
    assert result.runs == 3
    assert result.run_scores == [1.0, 0.5, 0.0]
    assert result.score == pytest.approx(0.5)
    assert result.score_std == pytest.approx(0.4082, abs=1e-3)


def test_repeat_aggregates_each_expectation():
    case = _case(CyclingAgent(["ab", "a", ""]), output_contains("a", "b", threshold=0.4))
    result = case.run(times=3)
    (e,) = result.expectations
    assert e.runs == 3
    assert e.score == pytest.approx(0.5)
    assert list(e.run_scores) == [1.0, 0.5, 0.0]
    assert e.score_std == pytest.approx(0.4082, abs=1e-3)
    assert e.passed  # mean 0.5 >= threshold 0.4


def test_repeat_mean_below_threshold_fails_even_if_one_run_passed():
    case = _case(CyclingAgent(["ab", "a", ""]), output_contains("a", "b", threshold=0.6))
    result = case.run(times=3)
    assert not result.expectations[0].passed
    assert not result.passed


def test_default_is_single_run():
    result = _case(fixed_agent("done"), output_equals("done")).run()
    assert result.runs == 1
    assert result.run_scores == []
    assert result.score_std == 0.0
    assert result.expectations[0].runs == 1


def test_repeat_builder_sets_times():
    case = _case(CyclingAgent(["x", "y"]), output_contains("x")).repeat(3)
    assert case.times == 3
    result = case.run()
    assert result.runs == 3


def test_run_times_argument_overrides_builder():
    case = _case(CyclingAgent(["x", "y"]), output_contains("x")).repeat(5)
    result = case.run(times=2)
    assert result.runs == 2


def test_repeat_validates_count():
    with pytest.raises(ValueError, match=">= 1"):
        new_case("x").repeat(0)


def test_partial_error_run_counts_as_zero():
    case = _case(FlakyAgent(fail_on={2}), output_equals("ok"))
    result = case.run(times=3)
    assert result.error is None
    assert result.run_scores == [1.0, 0.0, 1.0]
    assert result.score == pytest.approx(2 / 3)
    assert len(result.run_errors) == 1 and "boom" in result.run_errors[0]
    # expectations aggregate over the runs that produced them
    assert result.expectations[0].runs == 2
    assert result.expectations[0].score == 1.0


def test_all_runs_error_converges_to_case_error():
    case = _case(FlakyAgent(fail_on={1, 2, 3}), output_equals("ok"))
    result = case.run(times=3)
    assert result.error is not None and "boom" in result.error
    assert result.score == 0.0
    assert result.runs == 3
    assert not result.passed


def test_invalid_case_still_converges_without_repeating():
    # guardrail errors (missing agent) must not be executed N times
    result = new_case("orphan").when(text("x")).then(output_equals("y")).run(times=3)
    assert result.runs == 1
    assert "missing agent" in result.error


# ---------------------------------------------------------------------------
# Report rendering shows the spread
# ---------------------------------------------------------------------------


def test_text_report_shows_std_and_run_count():
    from smelt.report.text import render_text

    result = _case(CyclingAgent(["ab", "a", ""]), output_contains("a", "b")).run(times=3)
    out = render_text(result)
    assert "±" in out and "n=3" in out


def test_html_report_shows_std_and_run_count():
    from smelt.report.html import render_html

    result = _case(CyclingAgent(["ab", "a", ""]), output_contains("a", "b")).run(times=3)
    out = render_html(result)
    assert "±" in out and "3 runs" in out


def test_evaluate_markdown_shows_std_for_repeated_cases():
    case = _case(CyclingAgent(["ab", "a", ""]), output_contains("a", "b")).repeat(3)
    evaluation = (
        evaluate_skill(GOOD)
        .with_cases(case)
        .with_lint(False)
        .with_writing(enabled=False)
        .with_suggestions(enabled=False)
        .run()
    )
    md = evaluation.to_markdown()
    assert "±" in md and "n=3" in md


def test_case_result_json_includes_run_stats():
    case = _case(CyclingAgent(["ab", "a", ""]), output_contains("a", "b")).repeat(3)
    evaluation = (
        evaluate_skill(GOOD)
        .with_cases(case)
        .with_lint(False)
        .with_writing(enabled=False)
        .with_suggestions(enabled=False)
        .run()
    )
    import json

    payload = json.loads(evaluation.to_json())
    entry = payload["behavior"][0]
    assert entry["runs"] == 3
    assert entry["run_scores"] == [1.0, 0.5, 0.0]
    assert entry["expectations"][0]["score_std"] == pytest.approx(0.4082, abs=1e-3)


# ---------------------------------------------------------------------------
# pass^k reliability: a case is reliable only if EVERY run passed
# ---------------------------------------------------------------------------


def test_pass_hat_true_only_when_every_run_passed():
    # scores 1.0 / 1.0 / 0.0 against output_contains("a"): runs 1-2 pass, run 3 fails
    case = _case(CyclingAgent(["ab", "a", ""]), output_contains("a"))
    result = case.run(times=3)
    assert result.run_passed == [True, True, False]
    assert result.pass_hat is False  # pass^3: not reliable even though mean is 0.67

    reliable = _case(fixed_agent("done"), output_equals("done")).run(times=3)
    assert reliable.run_passed == [True, True, True]
    assert reliable.pass_hat is True


def test_pass_hat_error_run_counts_as_not_passed():
    case = _case(FlakyAgent(fail_on={2}), output_equals("ok"))
    result = case.run(times=3)
    assert result.run_passed == [True, False, True]
    assert result.pass_hat is False


def test_summary_includes_pass_hat_and_run_errors():
    case = _case(FlakyAgent(fail_on={2}), output_equals("ok"))
    summary = case.run(times=3).summary()
    assert "pass^3 ✘" in summary
    assert "run error" in summary and "boom" in summary


def test_pass_hat_none_for_single_run():
    result = _case(fixed_agent("done"), output_equals("done")).run()
    assert result.pass_hat is None


def test_text_report_shows_pass_hat():
    from smelt.report.text import render_text

    result = _case(CyclingAgent(["ab", "a", ""]), output_contains("a")).run(times=3)
    assert "pass^3 ✘" in render_text(result)


def test_html_report_shows_pass_hat():
    from smelt.report.html import render_html

    result = _case(fixed_agent("done"), output_equals("done")).run(times=3)
    assert "pass^3 ✔" in render_html(result)


def test_evaluate_markdown_and_json_include_pass_hat():
    case = _case(CyclingAgent(["done", "done", "nope"]), output_equals("done")).repeat(3)
    evaluation = (
        evaluate_skill(GOOD)
        .with_cases(case)
        .with_lint(False)
        .with_writing(enabled=False)
        .with_suggestions(enabled=False)
        .run()
    )
    md = evaluation.to_markdown()
    assert "pass^3 ✘" in md

    import json

    entry = json.loads(evaluation.to_json())["behavior"][0]
    assert entry["run_passed"] == [True, True, False]
    assert entry["pass_hat"] is False


# ---------------------------------------------------------------------------
# evaluate_skill: default repetition with deterministic degradation
# ---------------------------------------------------------------------------


def _quiet(builder):
    return builder.with_lint(False).with_writing(enabled=False).with_suggestions(enabled=False)


def test_evaluate_repeats_stochastic_cases_three_times_by_default():
    case = _case(CyclingAgent(["done"]), output_equals("done"))
    evaluation = _quiet(evaluate_skill(GOOD)).with_cases(case).run()
    assert evaluation.behavior_results[0].runs == 3


def test_evaluate_runs_deterministic_agents_once():
    fixed = _case(fixed_agent("done"), output_equals("done"), name="fixed")
    scripted = (
        new_case("scripted")
        .given(smelt_agent(llm=ScriptedLLM([LLMResponse.say("done")]), system_prompt="test"))
        .when(text("go"))
        .then(output_equals("done"))
    )
    evaluation = _quiet(evaluate_skill(GOOD)).with_cases(fixed, scripted).run()
    assert evaluation.behavior_results[0].runs == 1
    assert evaluation.behavior_results[1].runs == 1


def test_evaluate_with_times_overrides_auto_detection():
    case = _case(fixed_agent("done"), output_equals("done"))
    evaluation = _quiet(evaluate_skill(GOOD)).with_cases(case).with_times(2).run()
    assert evaluation.behavior_results[0].runs == 2


def test_evaluate_case_repeat_wins_over_auto_default():
    case = _case(CyclingAgent(["done"]), output_equals("done")).repeat(5)
    evaluation = _quiet(evaluate_skill(GOOD)).with_cases(case).run()
    assert evaluation.behavior_results[0].runs == 5


def test_evaluate_unbound_scripted_case_degrades_to_single_run():
    agent_llm = ScriptedLLM([LLMResponse.say("done")])
    case = new_case("auto").when(text("go")).then(output_equals("done"))
    evaluation = _quiet(evaluate_skill(GOOD, agent_llm=agent_llm)).with_cases(case).run()
    assert evaluation.behavior_results[0].runs == 1
