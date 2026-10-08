"""Tests for the evaluate module: comprehensive review, writing assessment,
suggestion generation, report rendering, and the CLI."""

import json
from pathlib import Path

import pytest

from smelt import (
    LLMResponse,
    ScriptedLLM,
    evaluate_skill,
    fixed_agent,
    new_case,
    output_equals,
    text,
    tool_call,
)
from smelt.evaluate import (
    DEFAULT_DIMENSIONS,
    SkillEvaluation,
    WritingAssessment,
    WritingDimension,
    _build_evidence,
)
from smelt.lint.models import Severity

ROOT = Path(__file__).resolve().parent.parent
GOOD = ROOT / "examples" / "good-skill"
BAD = ROOT / "examples" / "bad_skill"

WRITING_JSON = json.dumps({
    "dimensions": [
        {"name": "Metadata & naming", "score": 0.9, "comment": "accurate summary"},
        {"name": "Trigger guidance", "score": 0.4, "comment": "trigger scenarios underspecified"},
    ],
    "overall_comment": "usable overall; trigger guidance needs work",
}, ensure_ascii=False)

SUGGESTIONS_JSON = json.dumps({
    "suggestions": ["add trigger-word examples", "document edge cases", "trim section two"]
}, ensure_ascii=False)


def _judge(*responses: str) -> ScriptedLLM:
    return ScriptedLLM(list(responses))


def _passing_case():
    return (
        new_case("c1")
        .given(fixed_agent("done"))
        .when(text("go"))
        .then(output_equals("done"))
    )


# ---------------------------------------------------------------------------
# Building and running
# ---------------------------------------------------------------------------


def test_full_evaluation_pipeline(tmp_path):
    judge = _judge(WRITING_JSON, SUGGESTIONS_JSON)
    evaluation = (
        evaluate_skill(GOOD, judge=judge)
        .with_cases(_passing_case())
        .with_suggestions(max_items=2)
        .run()
    )
    assert evaluation.skill_name == "good-skill"
    assert evaluation.behavior_score == 1.0
    assert evaluation.lint_score == 1.0
    assert evaluation.writing_score == pytest.approx(0.65)
    # suggestions trimmed to max_items
    assert evaluation.suggestions == ["add trigger-word examples", "document edge cases"]
    # overall = (1.0*0.5 + 0.65*0.3 + 1.0*0.2) / 1.0 * 100
    assert evaluation.overall_score == pytest.approx(89.5)
    assert evaluation.grade == "B"


def test_unbound_case_auto_binds_to_skill():
    # a case without an agent: auto-bound to the skill under review with agent_llm
    agent_llm = ScriptedLLM([LLMResponse.call("run_command", {"cmd": "ls"}), LLMResponse.say("done")])
    case = (
        new_case("auto")
        .when(text("list the directory"))
        .then(tool_call("run_command"))
    )
    evaluation = (
        evaluate_skill(GOOD, judge=_judge(WRITING_JSON, SUGGESTIONS_JSON), agent_llm=agent_llm)
        .with_cases(case)
        .run()
    )
    assert evaluation.behavior_results[0].passed, evaluation.behavior_results[0].error
    assert evaluation.behavior_results[0].trace.tool_calls[0].name == "run_command"


def test_unbound_case_without_any_llm_raises():
    case = new_case("orphan").when(text("x")).then(output_equals("y"))
    with pytest.raises(ValueError, match="has no agent"):
        evaluate_skill(GOOD).with_cases(case).run()


def test_disabled_parts_excluded_from_overall():
    evaluation = (
        evaluate_skill(GOOD)  # no judge
        .with_cases(_passing_case())
        .with_lint(False)
        .run()
    )
    assert evaluation.lint_score is None
    assert evaluation.writing and evaluation.writing.error == "no judge LLM provided; writing review skipped"
    assert evaluation.writing_score is None
    assert evaluation.suggestions_error == "no judge LLM provided; suggestion generation skipped"
    # only the behavior part contributes
    assert evaluation.overall_score == pytest.approx(100.0)
    assert evaluation.grade == "A"


def test_no_data_at_all_overall_is_none():
    evaluation = evaluate_skill(GOOD).with_lint(False).with_writing(enabled=False).with_suggestions(enabled=False).run()
    assert evaluation.overall_score is None and evaluation.grade == "-"


def test_builder_immutability_and_validation():
    base = evaluate_skill(GOOD)
    derived = base.with_cases(_passing_case())
    assert base.cases == () and len(derived.cases) == 1

    with pytest.raises(ValueError, match="must not be empty"):
        base.with_writing(dimensions=[])
    with pytest.raises(ValueError, match="must not be negative"):
        base.with_weights(behavior=-1, writing=0.5, lint=0.2)

    custom = base.with_writing(dimensions=["just one dimension"])
    assert custom.writing_dimensions == ("just one dimension",)
    assert base.writing_dimensions == DEFAULT_DIMENSIONS


# ---------------------------------------------------------------------------
# Fault tolerance of writing review and suggestions
# ---------------------------------------------------------------------------


def test_writing_unparseable_judge_output():
    evaluation = evaluate_skill(GOOD, judge=_judge("definitely not json", SUGGESTIONS_JSON)).run()
    assert evaluation.writing is not None
    assert evaluation.writing.error is not None
    assert evaluation.writing_score is None
    # evidence is empty (good-skill is lint-clean, no behavior cases, writing
    # review failed) — the suggestions judge is never called
    assert evaluation.suggestions == []
    assert evaluation.suggestions_error is None


def test_writing_score_clamped_and_empty_dimensions():
    judge = _judge('{"dimensions": [{"name": "d", "score": 1.7}], "overall_comment": "x"}', SUGGESTIONS_JSON)
    evaluation = evaluate_skill(GOOD, judge=judge).run()
    assert evaluation.writing.dimensions[0].score == 1.0  # out-of-range clamped

    judge2 = _judge('{"dimensions": []}', SUGGESTIONS_JSON)
    evaluation2 = evaluate_skill(GOOD, judge=judge2).run()
    assert "empty dimensions" in evaluation2.writing.error


def test_suggestions_failure_recorded():
    judge = _judge(WRITING_JSON, "garbage output")
    evaluation = evaluate_skill(GOOD, judge=judge).run()
    assert evaluation.suggestions == []
    assert evaluation.suggestions_error is not None


def test_evidence_building_covers_all_parts():
    case_result = (
        new_case("f")
        .given(fixed_agent("wrong"))
        .when(text("go"))
        .then(output_equals("right"))
        .run()
    )
    evaluation = SkillEvaluation(
        skill_path=str(BAD),
        skill_name="bad_skill",
        behavior_results=[case_result],
        writing=WritingAssessment(
            dimensions=(WritingDimension(name="Trigger guidance", score=0.3, comment="weak"),),
        ),
        weights={"behavior": 0.5, "writing": 0.3, "lint": 0.2},
    )
    evidence = json.loads(_build_evidence(evaluation))
    assert evidence["behavior_tests"][0]["failed_expectations"]
    assert evidence["weak_writing_dimensions"][0]["dimension"] == "Trigger guidance"


# ---------------------------------------------------------------------------
# Report rendering and saving
# ---------------------------------------------------------------------------


def _full_evaluation() -> SkillEvaluation:
    judge = _judge(WRITING_JSON, SUGGESTIONS_JSON)
    return evaluate_skill(GOOD, judge=judge).with_cases(_passing_case()).run()


def test_markdown_report_sections():
    md = _full_evaluation().to_markdown()
    assert "# Skill Evaluation Report: good-skill" in md
    assert "## Behavior Tests" in md and "✅ c1" in md
    assert "## Static Lint" in md and "| Check |" in md
    assert "## Writing Review (LLM)" in md and "trigger guidance needs work" in md
    assert "## Improvement Suggestions" in md and "1. add trigger-word examples" in md
    assert "Overall: **100.0 / 100**" not in md  # the total is the weighted 89.5
    assert "89.5" in md


def test_markdown_with_error_and_disabled_parts():
    evaluation = evaluate_skill(GOOD).with_lint(False).run()
    md = evaluation.to_markdown()
    assert "Review failed" in md or "writing review skipped" in md
    assert "Suggestion generation failed: no judge LLM" in md
    assert "## Static Lint" not in md


def test_json_report_structure():
    payload = json.loads(_full_evaluation().to_json())
    assert payload["skill"]["name"] == "good-skill"
    assert payload["overall"]["grade"] == "B"
    assert payload["scores"]["behavior"] == 1.0
    assert payload["behavior"][0]["expectations"][0]["passed"] is True
    assert payload["lint"]["checks"]
    assert payload["writing"]["dimensions"][0]["name"] == "Metadata & naming"
    assert len(payload["suggestions"]) == 3


def test_save_by_extension(tmp_path):
    evaluation = _full_evaluation()
    md_path = evaluation.save(tmp_path / "reports" / "eval.md")
    assert md_path.read_text(encoding="utf-8").startswith("# Skill Evaluation Report")

    json_path = evaluation.save(tmp_path / "eval.json")
    assert json.loads(json_path.read_text(encoding="utf-8"))["skill"]["name"] == "good-skill"

    explicit = evaluation.save(tmp_path / "eval.txt", format="json")
    assert explicit.read_text(encoding="utf-8").startswith("{")


# ---------------------------------------------------------------------------
# CLI evaluate
# ---------------------------------------------------------------------------


def test_cli_evaluate_lint_only_prints_markdown(capsys):
    from smelt.cli import main

    code = main(["evaluate", str(GOOD), "--no-writing", "--no-suggestions"])
    out = capsys.readouterr().out
    assert code == 0
    assert "# Skill Evaluation Report: good-skill" in out
    assert "Overall: **100.0 / 100**" in out


def test_cli_evaluate_without_judge_degrades_gracefully(capsys):
    from smelt.cli import main

    code = main(["evaluate", str(BAD)])
    out, err = capsys.readouterr()
    assert "skipped" in err  # note that writing/suggestions are skipped
    assert code == 1  # bad_skill's lint score is below the default fail-under
    assert "bad_skill" in out


def test_cli_evaluate_output_and_fail_under(tmp_path, capsys):
    from smelt.cli import main

    report = tmp_path / "r.json"
    code = main([
        "evaluate", str(GOOD),
        "--no-writing", "--no-suggestions",
        "--output", str(report),
        "--fail-under", "95",
    ])
    assert code == 0
    assert f"report written to {report}" in capsys.readouterr().out
    assert json.loads(report.read_text(encoding="utf-8"))["overall"]["score"] == 100.0


def test_cli_evaluate_bad_path_and_bad_cases(tmp_path, capsys):
    from smelt.cli import main

    assert main(["evaluate", str(tmp_path / "ghost"), "--no-writing", "--no-suggestions"]) == 2
    assert "evaluation failed" in capsys.readouterr().err

    bad_cases = tmp_path / "bad.py"
    bad_cases.write_text("raise RuntimeError('x')", encoding="utf-8")
    assert main(["evaluate", str(GOOD), "--cases", str(bad_cases), "--no-writing", "--no-suggestions"]) == 2
    assert "failed to load cases" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Code-first suggestions
# ---------------------------------------------------------------------------


def test_code_suggestions_from_lint_without_judge():
    evaluation = evaluate_skill(BAD).with_suggestions().run()
    assert evaluation.suggestions  # non-empty without any judge
    assert evaluation.suggestions_error is None
    assert all(s.startswith("[") and " → " in s for s in evaluation.suggestions)


def test_code_suggestions_errors_before_warnings():
    evaluation = evaluate_skill(BAD).with_suggestions().run()
    report = evaluation.lint_report
    severity_of = {m.text: m.severity for c in report.results for m in c.messages if m.fix}
    texts = [s.split("] ", 1)[1].split(" → ")[0] for s in evaluation.suggestions]
    severities = [severity_of[t] for t in texts]
    first_non_error = next(
        (i for i, s in enumerate(severities) if s is not Severity.ERROR),
        len(severities),
    )
    assert all(s is Severity.ERROR for s in severities[:first_non_error])
    assert all(s is not Severity.ERROR for s in severities[first_non_error:])


def test_code_suggestions_precede_llm_suggestions():
    judge = _judge(WRITING_JSON, SUGGESTIONS_JSON)
    evaluation = evaluate_skill(BAD, judge=judge).with_suggestions(max_items=20).run()
    code_prefix = [s for s in evaluation.suggestions if s.startswith("[")]
    assert code_prefix == evaluation.suggestions[: len(code_prefix)]
    assert any(not s.startswith("[") for s in evaluation.suggestions)  # LLM tail


def test_evidence_no_longer_contains_lint_issues():
    evaluation = evaluate_skill(BAD).run()
    import json as _json

    assert "lint_issues" not in _json.loads(_build_evidence(evaluation) or "{}")


def test_judge_skipped_when_code_suggestions_fill_max():
    judge = _judge(WRITING_JSON, SUGGESTIONS_JSON)
    evaluation = evaluate_skill(BAD, judge=judge).with_suggestions(max_items=1).run()
    assert len(evaluation.suggestions) == 1
    assert len(judge.calls) == 1  # only the writing review call


def test_markdown_empty_state_when_nothing_to_suggest():
    evaluation = evaluate_skill(GOOD).run()  # no judge, clean skill
    assert evaluation.suggestions == []
    assert "nothing to suggest" in evaluation.to_markdown()


def test_markdown_not_enabled_state_preserved():
    evaluation = evaluate_skill(GOOD).with_suggestions(enabled=False).run()
    assert "(not enabled)" in evaluation.to_markdown()
