"""tests/test_canary.py"""
from smelt import LLMResponse, ScriptedLLM
from smelt.challenge.canary import run_canary


def test_miscalibrated_judge_scores_canary_high():
    judge = ScriptedLLM([LLMResponse.say('{"reason": "fluent answer", "score": 0.9}')])
    result = run_canary(judge)
    assert not result.calibrated
    assert result.score == 0.9


def test_calibrated_judge_scores_canary_low():
    judge = ScriptedLLM([LLMResponse.say('{"reason": "off-topic and invented tool result", "score": 0.1}')])
    result = run_canary(judge)
    assert result.calibrated
    assert result.score == 0.1


def test_judge_failure_counts_as_miscalibrated():
    judge = ScriptedLLM([LLMResponse.say("not json at all")])
    result = run_canary(judge)
    # judge parse failures score 0 but prove nothing → flagged via the message
    assert not result.calibrated
