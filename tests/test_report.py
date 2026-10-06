"""Tests for the report package and the .report() / .report_cli() chain tails."""

from pathlib import Path

from smelt import (
    LLMResponse,
    ScriptedLLM,
    fixed_agent,
    llm,
    new_case,
    output_contains,
    output_equals,
    smelt_agent,
    text,
    tool_call,
)
from smelt.report import render_html, render_text, write_html_report
from smelt.results import CaseResult, ExpectationResult
from smelt.trace import ToolCallRecord, Trace


def _passing_case():
    return (
        new_case("report-pass")
        .given(fixed_agent("all done", tool_calls=[{"name": "run_command", "arguments": {"cmd": "ls"}}]))
        .when(text("go"))
        .then(tool_call("run_command"))
        .then(output_contains("done"))
    )


def _failing_result() -> CaseResult:
    return (
        new_case("report-fail")
        .given(fixed_agent("partial"))
        .when(text("go"))
        .then(output_equals("expected something else"))
        .run()
    )


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------


def test_render_html_pass_and_fail():
    ok_html = render_html(_passing_case().run())
    assert "PASS" in ok_html and "report-pass" in ok_html
    assert "tool_call(run_command)" in ok_html
    assert "run_command" in ok_html  # trace tool table

    fail_html = render_html(_failing_result())
    assert "FAIL" in fail_html and "output_equals" in fail_html


def test_render_html_escapes_user_content():
    trace = Trace(output="<script>alert(1)</script>")
    result = CaseResult(
        case_name="xss <b>",
        expectations=[ExpectationResult(name="e", score=0.0, threshold=1.0, message="<img src=x>")],
        trace=trace,
        error=None,
    )
    html_doc = render_html(result)
    assert "<script>alert(1)</script>" not in html_doc
    assert "&lt;script&gt;" in html_doc
    assert "&lt;img src=x&gt;" in html_doc


def test_render_html_error_result_without_trace():
    result = CaseResult(case_name="err", expectations=[], error="missing agent: ...")
    html_doc = render_html(result)
    assert "FAIL" in html_doc and "missing agent" in html_doc
    assert "No expectations" in html_doc


# ---------------------------------------------------------------------------
# Terminal text rendering
# ---------------------------------------------------------------------------


def test_render_text_sections():
    out = render_text(_passing_case().run())
    assert "SMELT REPORT" in out and "report-pass" in out
    assert "PASS" in out and "score" in out
    assert "expectations" in out and "✔" in out
    assert "trace" in out and "run_command" in out
    assert "final output" in out and "all done" in out


def test_render_text_failure_and_error_paths():
    out = render_text(_failing_result())
    assert "FAIL" in out and "✘" in out and "actual output" in out

    err = CaseResult(case_name="e", expectations=[], error="missing agent: attach one")
    out2 = render_text(err)
    assert "error:" in out2 and "(none — flow-only run)" in out2


def test_render_text_tool_error_and_long_args():
    trace = Trace(
        output="",
        tool_calls=[ToolCallRecord(name="t", arguments={"long": "x" * 100}, error="boom")],
    )
    out = render_text(CaseResult(case_name="c", expectations=[], trace=trace))
    assert "boom" in out and "..." in out and "(empty)" in out


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


def test_write_html_report_creates_timestamped_and_latest(tmp_path):
    result = _passing_case().run()
    path = write_html_report(result, tmp_path / "reports")
    assert path.name.startswith("report-pass-") and path.suffix == ".html"
    assert path.exists()
    latest = tmp_path / "reports" / "report-pass-latest.html"
    assert latest.exists()
    assert "PASS" in latest.read_text(encoding="utf-8")


def test_write_html_report_slugifies_names(tmp_path):
    result = CaseResult(case_name=" spaces/危险 ", expectations=[])
    path = write_html_report(result, tmp_path)
    assert path.name.startswith("spaces-")


# ---------------------------------------------------------------------------
# Chain tails on SmeltCase
# ---------------------------------------------------------------------------


def test_case_report_writes_html_and_returns_result(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    result = _passing_case().report()
    assert result.passed
    assert result.report_path is not None
    report = tmp_path / result.report_path  # report_path is relative to the project root
    assert report.parent == tmp_path / ".smelt" / "reports"
    assert "PASS" in report.read_text(encoding="utf-8")
    assert "report written to" in capsys.readouterr().out
    # still pytest-compatible
    assert result.assert_passed() is result


def test_case_report_custom_output_dir_and_quiet(tmp_path, capsys):
    result = _passing_case().report(output_dir=str(tmp_path / "custom"), quiet=True)
    assert Path(result.report_path).parent == tmp_path / "custom"
    assert capsys.readouterr().out == ""


def test_case_report_on_invalid_case_still_writes_error_report(tmp_path):
    result = new_case("invalid").when(text("go")).report(output_dir=str(tmp_path))
    assert not result.passed and "missing agent" in result.error
    assert "missing agent" in Path(result.report_path).read_text(encoding="utf-8")


def test_case_report_cli_prints_formatted_report(capsys):
    result = _passing_case().report_cli()
    out = capsys.readouterr().out
    assert result.passed
    assert "SMELT REPORT" in out and "✔" in out and "═" in out


def test_case_report_cli_quiet(capsys):
    result = _failing_result_case().report_cli(quiet=True)
    assert not result.passed
    assert capsys.readouterr().out == ""


def _failing_result_case():
    return new_case("quiet-fail").given(fixed_agent("x")).when(text("go")).then(output_equals("y"))


def test_chain_tail_with_fragmented_given():
    """Fragments entry works with the report tails too."""
    case = (
        smelt_agent.new_case("frag-report")
        .given(llm(ScriptedLLM([LLMResponse.say("fine")])))
        .when(text("go"))
        .then(output_equals("fine"))
    )
    cli_result = case.report_cli(quiet=True)
    assert cli_result.passed
