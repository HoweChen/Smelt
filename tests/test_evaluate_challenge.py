"""tests/test_evaluate_challenge.py"""
from pathlib import Path

from smelt import LLMResponse, ScriptedLLM, evaluate_skill, new_case, text, tool_call

SKILL_MD = "---\nname: s\n---\n\n# S\n\n## B\nDo not commit merges.\n"


def make_skill(tmp_path: Path) -> Path:
    d = tmp_path / "skill"
    d.mkdir()
    (d / "SKILL.md").write_text(SKILL_MD, encoding="utf-8")
    return d


def scripted_judge(score: float = 1.0):
    return ScriptedLLM([LLMResponse.say(f'{{"reason": "r", "score": {score}}}')])


def test_challenge_runs_by_default(tmp_path):
    challenger = ScriptedLLM([LLMResponse.say(
        '{"probes": [{"type": "no-trigger", "trigger": "hi", "expectation": "no action", "fixture": null}]}'
    )])
    report = (
        evaluate_skill(make_skill(tmp_path), judge=scripted_judge(), challenger=challenger)
        .with_writing(enabled=False)
        .with_suggestions(enabled=False)
        .run()
    )
    assert report.challenge is not None
    assert report.challenge.skipped is None
    assert len(report.challenge.probes) == 1


def test_challenge_opt_out(tmp_path):
    report = (
        evaluate_skill(make_skill(tmp_path), judge=scripted_judge())
        .with_writing(enabled=False)
        .with_suggestions(enabled=False)
        .with_challenge(enabled=False)
        .run()
    )
    assert report.challenge is None


def test_challenge_skips_without_challenger(tmp_path):
    builder = evaluate_skill(make_skill(tmp_path))  # no judge, no challenger
    builder = builder.with_writing(enabled=False).with_suggestions(enabled=False)
    report = builder.run()
    assert report.challenge is not None
    assert report.challenge.skipped is not None
    assert "no challenger" in report.challenge.skipped


def test_challenge_skips_deterministic_backend(tmp_path):
    case = new_case("c").when(text("hi")).then(tool_call("run_command"))
    report = (
        evaluate_skill(make_skill(tmp_path), judge=scripted_judge())
        .with_writing(enabled=False).with_suggestions(enabled=False)
        .with_cases(case)
        .run()
    )
    # auto-bound cases use the judge (a ScriptedLLM) → deterministic → skipped
    assert report.challenge is not None and report.challenge.skipped == "deterministic backend"


def test_challenge_not_in_overall_score(tmp_path):
    challenger = ScriptedLLM([LLMResponse.say(
        '{"probes": [{"type": "no-trigger", "trigger": "hi", "expectation": "no action", "fixture": null}]}'
    )])
    report = (
        evaluate_skill(make_skill(tmp_path), judge=scripted_judge(0.0), challenger=challenger)
        .with_writing(enabled=False).with_suggestions(enabled=False)
        .run()
    )
    # judge scores the probe 0 → break → challenge score 0, but overall is unchanged
    assert report.challenge.score == 0.0
    assert "challenge" not in report.weights


def test_with_weights_accepts_challenge():
    from smelt.evaluate import SkillEvaluationBuilder
    b = SkillEvaluationBuilder(skill_path=Path("x")).with_weights(behavior=0.4, writing=0.2, lint=0.2, challenge=0.2)
    assert b.weight_map["challenge"] == 0.2
