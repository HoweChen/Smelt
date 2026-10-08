"""Tests for compare(): version-to-version diff of two skill evaluations.

Significance rule: a score difference counts only when it exceeds the noise
band max(min_delta, 2σ) where σ combines both sides' run spreads. A pass^k
drop (True → False) is a reliability regression regardless of the mean.
"""

import json
from pathlib import Path

import pytest

from smelt import compare, evaluate_skill, fixed_agent, new_case, output_equals, text
from smelt.compare import _as_payload

ROOT = Path(__file__).resolve().parent.parent
GOOD = ROOT / "examples" / "good-skill"


def _payload(cases, overall=80.0, name="skill"):
    """Minimal evaluate-JSON-shaped payload for one version."""
    return {
        "skill": {"path": "/x", "name": name},
        "overall": {"score": overall, "grade": "B", "weights": {}},
        "behavior": [
            {
                "case": c["case"],
                "score": c["score"],
                "passed": all(p >= 0.5 for p in c.get("run_passed", [True])) if c.get("run_passed") else True,
                "error": c.get("error"),
                "runs": c.get("runs", 1),
                "run_scores": c.get("run_scores", []),
                "run_passed": c.get("run_passed", []),
                "pass_hat": c.get("pass_hat"),
                "score_std": c.get("score_std", 0.0),
                "expectations": [],
            }
            for c in cases
        ],
    }


# ---------------------------------------------------------------------------
# Significance: diff vs noise band
# ---------------------------------------------------------------------------


def test_compare_improved_regressed_unchanged():
    baseline = _payload([
        {"case": "A", "score": 0.8, "score_std": 0.02, "runs": 3},
        {"case": "B", "score": 0.8, "score_std": 0.02, "runs": 3},
        {"case": "C", "score": 0.9, "score_std": 0.02, "runs": 3},
    ])
    candidate = _payload([
        {"case": "A", "score": 0.9, "score_std": 0.02, "runs": 3},   # +0.10 > band → improved
        {"case": "B", "score": 0.83, "score_std": 0.02, "runs": 3},  # +0.03 < band → unchanged
        {"case": "C", "score": 0.6, "score_std": 0.02, "runs": 3},   # -0.30 → regressed
    ])
    result = compare(baseline, candidate)
    verdicts = {d.case: d.verdict for d in result.cases}
    assert verdicts == {"A": "improved", "B": "unchanged", "C": "regressed"}
    assert result.regressions[0].case == "C"
    assert result.improvements[0].case == "A"
    assert result.has_regression


def test_compare_single_runs_fall_back_to_min_delta():
    baseline = _payload([{"case": "A", "score": 0.80}, {"case": "B", "score": 0.80}])
    candidate = _payload([{"case": "A", "score": 0.84}, {"case": "B", "score": 0.86}])
    result = compare(baseline, candidate, min_delta=0.05)
    verdicts = {d.case: d.verdict for d in result.cases}
    assert verdicts == {"A": "unchanged", "B": "improved"}


def test_compare_pass_hat_drop_is_reliability_regression():
    baseline = _payload([{
        "case": "A", "score": 0.8, "runs": 3,
        "run_passed": [True, True, True], "pass_hat": True, "score_std": 0.05,
    }])
    candidate = _payload([{
        "case": "A", "score": 0.82, "runs": 3,  # mean even slightly up...
        "run_passed": [True, True, False], "pass_hat": False, "score_std": 0.15,
    }])
    result = compare(baseline, candidate)
    (d,) = result.cases
    assert d.verdict == "regressed"
    assert "reliability" in d.reason
    assert result.has_regression


def test_compare_pass_hat_recovery_is_improvement():
    baseline = _payload([{
        "case": "A", "score": 0.8, "runs": 3, "run_passed": [True, False, True], "pass_hat": False,
    }])
    candidate = _payload([{
        "case": "A", "score": 0.82, "runs": 3, "run_passed": [True, True, True], "pass_hat": True,
    }])
    (d,) = compare(baseline, candidate).cases
    assert d.verdict == "improved"
    assert "reliability" in d.reason


def test_compare_added_removed_and_error_cases():
    baseline = _payload([{"case": "kept", "score": 0.8}, {"case": "dropped", "score": 0.7}])
    candidate = _payload([
        {"case": "kept", "score": 0.8},
        {"case": "new", "score": 0.9},
        {"case": "broken", "score": 0.0, "error": "RuntimeError: boom"},
    ])
    result = compare(baseline, candidate)
    verdicts = {d.case: d.verdict for d in result.cases}
    assert verdicts["dropped"] == "removed"
    assert verdicts["new"] == "added"
    assert verdicts["broken"] == "added"  # no baseline to regress against
    # an error in the CANDIDATE on a case present in BOTH is a regression
    candidate2 = _payload([
        {"case": "kept", "score": 0.0, "error": "RuntimeError: boom"},
    ])
    verdicts2 = {d.case: d for d in compare(baseline, candidate2).cases}
    d = verdicts2["kept"]
    assert d.verdict == "regressed" and "error" in d.reason


def test_compare_min_delta_validated():
    with pytest.raises(ValueError, match="min_delta"):
        compare(_payload([]), _payload([]), min_delta=-1)


# ---------------------------------------------------------------------------
# Whole-report level
# ---------------------------------------------------------------------------


def test_compare_overall_diff_and_assert_no_regression():
    baseline = _payload([{"case": "A", "score": 0.9}], overall=85.0, name="v1")
    candidate = _payload([{"case": "A", "score": 0.5}], overall=70.0, name="v2")
    result = compare(baseline, candidate)
    assert result.overall_diff == pytest.approx(-15.0)
    assert result.baseline_name == "v1" and result.candidate_name == "v2"
    with pytest.raises(AssertionError, match="regression"):
        result.assert_no_regression()

    same = compare(baseline, baseline)
    assert same.overall_diff == 0.0
    assert not same.has_regression
    same.assert_no_regression()  # no raise


def test_compare_markdown_report():
    baseline = _payload([{"case": "A", "score": 0.9, "runs": 3, "score_std": 0.02}], overall=85.0, name="v1")
    candidate = _payload([{"case": "A", "score": 0.6, "runs": 3, "score_std": 0.02}], overall=70.0, name="v2")
    md = compare(baseline, candidate).to_markdown()
    assert "# Skill Comparison" in md
    assert "v1" in md and "v2" in md
    assert "regressed" in md
    assert "-15.0" in md


def test_compare_result_to_dict_roundtrip():
    result = compare(_payload([{"case": "A", "score": 0.9}]), _payload([{"case": "A", "score": 0.5}]))
    payload = json.loads(json.dumps(result.to_dict()))  # JSON-serializable
    assert payload["cases"][0]["case"] == "A"
    assert payload["cases"][0]["verdict"] == "regressed"
    assert payload["has_regression"] is True


# ---------------------------------------------------------------------------
# Inputs: SkillEvaluation objects and saved JSON reports
# ---------------------------------------------------------------------------


def _eval(tmp_path, name, output):
    case = new_case(name).given(fixed_agent(output)).when(text("go")).then(output_equals("done"))
    return (
        evaluate_skill(GOOD)
        .with_cases(case)
        .with_lint(False)
        .with_writing(enabled=False)
        .with_suggestions(enabled=False)
        .with_times(1)
        .run()
    )


def test_compare_accepts_skill_evaluation_objects(tmp_path):
    v1 = _eval(tmp_path, "c", "done")
    v2 = _eval(tmp_path, "c", "nope")
    result = compare(v1, v2)
    assert result.cases[0].verdict == "regressed"


def test_compare_accepts_saved_json_paths(tmp_path):
    v1 = _eval(tmp_path, "c", "done")
    v2 = _eval(tmp_path, "c", "nope")
    p1 = v1.save(tmp_path / "v1.json")
    p2 = v2.save(tmp_path / "v2.json")
    result = compare(p1, p2)
    assert result.cases[0].verdict == "regressed"
    assert result.has_regression


def test_as_payload_rejects_unknown_type():
    with pytest.raises(TypeError, match="compare"):
        _as_payload(42)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_compare_exit_codes_and_output(tmp_path, capsys):
    from smelt.cli import main

    v1 = _eval(tmp_path, "c", "done")
    v2 = _eval(tmp_path, "c", "nope")
    p1 = v1.save(tmp_path / "v1.json")
    p2 = v2.save(tmp_path / "v2.json")

    assert main(["compare", str(p1), str(p2)]) == 1  # regression → non-zero
    out = capsys.readouterr().out
    assert "regressed" in out

    assert main(["compare", str(p1), str(p1)]) == 0  # identical → clean

    report = tmp_path / "diff.md"
    assert main(["compare", str(p1), str(p2), "--output", str(report)]) == 1
    assert "# Skill Comparison" in report.read_text(encoding="utf-8")


def test_cli_compare_bad_path(capsys):
    from smelt.cli import main

    assert main(["compare", "ghost-a.json", "ghost-b.json"]) == 2
    assert "compare failed" in capsys.readouterr().err
