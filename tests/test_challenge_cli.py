"""tests/test_challenge_cli.py"""
import json

from smelt.cli import main
from smelt.compare import compare


def test_compare_surfaces_challenge_note():
    base = {"skill": {"name": "s"}, "behavior": [],
            "challenge": {"skipped": None, "breaks": 2, "score": 0.5, "probes": []}}
    cand = {"skill": {"name": "s"}, "behavior": [],
            "challenge": {"skipped": None, "breaks": 0, "score": 1.0, "probes": []}}
    result = compare(base, cand)
    assert result.challenge_note == "challenge breaks: 2 → 0"
    assert not result.has_regression  # challenge never gates


def test_compare_note_absent_without_challenge():
    result = compare({"skill": {"name": "s"}, "behavior": []},
                     {"skill": {"name": "s"}, "behavior": []})
    assert result.challenge_note is None


def test_evaluate_no_challenge_flag(tmp_path, monkeypatch):
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: s\n---\n\n# S\n\nbody\n", encoding="utf-8")
    out = tmp_path / "r.json"
    for key in list(__import__("os").environ):
        if key.startswith("SMELT_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr("smelt.llm_config._auto_load", lambda: None)
    monkeypatch.setattr("smelt.env._auto_load", lambda: None)
    code = main(["evaluate", str(skill), "--no-challenge", "--no-writing",
                 "--no-suggestions", "--output", str(out)])
    assert code in (0, 1)  # no crash; score gate decides
    assert json.loads(out.read_text())["challenge"] is None
