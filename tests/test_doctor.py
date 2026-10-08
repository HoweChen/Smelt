"""tests/test_doctor.py"""
from pathlib import Path

from smelt import LLMResponse, ScriptedLLM
from smelt.challenge.doctor import doctor
from smelt.results import CaseResult, ExpectationResult

SKILL_MD = """---
name: commit
---

# Commit

## Boundaries
Do not commit merges.
"""


def make_skill(tmp_path: Path) -> Path:
    d = tmp_path / "skill"
    d.mkdir()
    (d / "SKILL.md").write_text(SKILL_MD, encoding="utf-8")
    return d


def make_case_file(tmp_path: Path) -> Path:
    f = tmp_path / "cases.py"
    f.write_text(
        "from smelt import new_case, text, tool_call\n"
        "c = new_case('c1').when(text('commit')).then(tool_call('run_command'))\n",
        encoding="utf-8",
    )
    return f


def score_run(score: float):
    def _run(cases, skill_dir, times=None):
        return [CaseResult(case_name=c.name,
                           expectations=[ExpectationResult("e", score, 0.5)]) for c in cases]
    return _run


def flat_judge():
    return ScriptedLLM([LLMResponse.say('{"reason": "ok", "score": 0.1}')])


def test_all_mutants_killed(tmp_path):
    # Baseline 1.0; every mutant dir scores 0.0 → all killed → score 1.0, ok.
    def _run(cases, skill_dir, times=None):
        score = 1.0 if skill_dir.name == "pristine" else 0.0
        return score_run(score)(cases, skill_dir, times)

    report = doctor([make_case_file(tmp_path)], skill=make_skill(tmp_path),
                    doctor=flat_judge(), _run=_run)
    assert report.mutation_score == 1.0
    assert report.ok


def test_surviving_mutant_reported(tmp_path):
    # Mutants never change the score → all survive → mutation score 0, not ok.
    report = doctor([make_case_file(tmp_path)], skill=make_skill(tmp_path),
                    doctor=flat_judge(), _run=score_run(1.0))
    assert report.mutation_score == 0.0
    assert not report.ok
    assert all(m.verdict == "survived" for m in report.mutants if m.verdict != "invalid")


def test_deterministic_cases_mark_mutation_na(tmp_path):
    f = tmp_path / "cases.py"
    f.write_text(
        "from smelt import new_case, fixed_agent, text\n"
        "c = new_case('fx').given(fixed_agent('done')).when(text('hi'))\n",
        encoding="utf-8",
    )
    report = doctor([f], skill=make_skill(tmp_path), doctor=flat_judge(), _run=score_run(1.0))
    assert report.mutation_score is None
    assert any("deterministic" in n for n in report.notes)


def test_canary_miscalibrated_marks_not_ok(tmp_path):
    bad_judge = ScriptedLLM([LLMResponse.say('{"reason": "great", "score": 0.95}')])
    report = doctor([make_case_file(tmp_path)], skill=make_skill(tmp_path),
                    doctor=bad_judge, _run=score_run(1.0))
    assert report.canary is not None and not report.canary.calibrated
    assert not report.ok


def test_red_baseline_cases_excluded_and_flagged(tmp_path):
    # Case fails against the UNMUTATED skill (score 0.0) → red baseline:
    # excluded from kill attribution, flagged, mutation verdicts are N/A.
    report = doctor([make_case_file(tmp_path)], skill=make_skill(tmp_path),
                    doctor=flat_judge(), _run=score_run(0.0))
    assert report.red_baselines == ["c1"]
    assert report.mutation_score is None
    assert any("baseline" in n for n in report.notes)
    assert not report.ok


def test_mixed_green_and_red_baselines(tmp_path):
    f = tmp_path / "cases.py"
    f.write_text(
        "from smelt import new_case, text, tool_call\n"
        "a = new_case('green').when(text('commit')).then(tool_call('run_command'))\n"
        "b = new_case('red').when(text('commit')).then(tool_call('run_command'))\n",
        encoding="utf-8",
    )

    def _run(cases, skill_dir, times=None):
        out = []
        for c in cases:
            score = 0.0 if c.name == "red" else (1.0 if skill_dir.name == "pristine" else 0.0)
            out.append(CaseResult(case_name=c.name, expectations=[ExpectationResult("e", score, 0.5)]))
        return out

    report = doctor([f], skill=make_skill(tmp_path), doctor=flat_judge(), _run=_run)
    assert report.red_baselines == ["red"]
    assert report.mutation_score == 1.0  # the green case kills every mutant
    assert not report.ok  # a red baseline is itself a health issue
