"""tests/test_probes.py"""
from pathlib import Path

from smelt import LLMResponse, ScriptedLLM
from smelt.challenge.probes import (
    Probe,
    challenge_skill,
    generate_probes,
    run_probe,
)

SKILL_MD = "---\nname: commit\n---\n\n# Commit\n\nCommit changes when asked.\n"


def make_skill(tmp_path: Path) -> Path:
    d = tmp_path / "skill"
    d.mkdir()
    (d / "SKILL.md").write_text(SKILL_MD, encoding="utf-8")
    return d


def test_generate_probes_parses_json():
    challenger = ScriptedLLM([LLMResponse.say(
        '{"probes": [{"type": "hard-trigger", "trigger": "tidy up my edits",'
        ' "expectation": "commits the changes", "fixture": null}]}'
    )])
    probes = generate_probes(challenger, SKILL_MD, max_probes=8)
    assert probes == [Probe(type="hard-trigger", trigger="tidy up my edits",
                            expectation="commits the changes", fixture=None)]


def test_generate_probes_bad_json_returns_empty():
    challenger = ScriptedLLM([LLMResponse.say("no json here")])
    assert generate_probes(challenger, SKILL_MD, max_probes=8) == []


def test_run_probe_break_when_judge_fails(tmp_path):
    probe = Probe(type="hard-trigger", trigger="tidy up", expectation="commits")
    agent = ScriptedLLM([LLMResponse.say("I did nothing")])
    judge = ScriptedLLM([LLMResponse.say('{"reason": "did not act", "score": 0.1}')])
    result = run_probe(probe, skill=make_skill(tmp_path), agent_llm=agent, judge=judge, tools=())
    assert result.verdict == "break"
    assert "new_case" in result.suggestion


def test_run_probe_survived_when_judge_passes(tmp_path):
    probe = Probe(type="no-trigger", trigger="what time is it", expectation="no git action")
    agent = ScriptedLLM([LLMResponse.say("It is noon.")])
    judge = ScriptedLLM([LLMResponse.say('{"reason": "correct restraint", "score": 1.0}')])
    result = run_probe(probe, skill=make_skill(tmp_path), agent_llm=agent, judge=judge, tools=())
    assert result.verdict == "survived"


def test_challenge_skill_aggregates(tmp_path):
    challenger = ScriptedLLM([LLMResponse.say(
        '{"probes": ['
        '{"type": "hard-trigger", "trigger": "t1", "expectation": "e1", "fixture": null},'
        '{"type": "no-trigger", "trigger": "t2", "expectation": "e2", "fixture": null}]}'
    )])
    # agent answers; judge fails t1, passes t2 → 1 break of 2
    agent = ScriptedLLM([LLMResponse.say("answer")])
    judge = ScriptedLLM([
        LLMResponse.say('{"reason": "fail", "score": 0.0}'),
        LLMResponse.say('{"reason": "pass", "score": 1.0}'),
    ])
    result = challenge_skill(make_skill(tmp_path), challenger=challenger,
                             agent_llm=agent, judge=judge)
    assert len(result.probes) == 2
    assert len(result.breaks) == 1
    assert result.score == 0.5
    assert result.skipped is None
