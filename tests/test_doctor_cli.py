"""tests/test_doctor_cli.py"""
import argparse
from pathlib import Path

from smelt.cli import main


def _setup(tmp_path: Path):
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: s\n---\n\n# S\n\n## B\nno merges\n", encoding="utf-8")
    cases = tmp_path / "cases.py"
    cases.write_text(
        "from smelt import new_case, fixed_agent, text\n"
        "c = new_case('fx').given(fixed_agent('done')).when(text('hi'))\n",
        encoding="utf-8",
    )
    return skill, cases


def test_missing_doctor_config_exits_2(tmp_path, monkeypatch, capsys):
    for key in list(__import__("os").environ):
        if key.startswith("SMELT_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr("smelt.llm_config._auto_load", lambda: None)
    skill, cases = _setup(tmp_path)
    code = main(["doctor", str(cases), "--skill", str(skill)])
    assert code == 2
    assert "doctor agent not configured" in capsys.readouterr().err


def test_doctor_ok_exit_0(tmp_path, capsys):
    from smelt import LLMResponse, ScriptedLLM

    judge = ScriptedLLM([LLMResponse.say('{"reason": "bad answer", "score": 0.1}')])
    skill, cases = _setup(tmp_path)
    from smelt.cli import _cmd_doctor

    args = argparse.Namespace(cases=[cases], skill=skill, min_score=0.8, output=None)
    code = _cmd_doctor(args, doctor_llm=judge)
    assert code == 0  # deterministic case → mutation N/A, canary calibrated → ok
    assert "doctor" in capsys.readouterr().out


def test_doctor_issues_exit_1(tmp_path, capsys):
    from smelt import LLMResponse, ScriptedLLM

    bad_judge = ScriptedLLM([LLMResponse.say('{"reason": "great", "score": 0.95}')])
    skill, cases = _setup(tmp_path)
    from smelt.cli import _cmd_doctor

    args = argparse.Namespace(cases=[cases], skill=skill, min_score=0.8, output=None)
    code = _cmd_doctor(args, doctor_llm=bad_judge)
    assert code == 1  # canary miscalibrated → issues found
