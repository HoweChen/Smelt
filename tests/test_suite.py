"""Tests for suite-level summary reports: index.html, per-case pages, terminal summary."""

from pathlib import Path

import pytest

from smelt import (
    fixed_agent,
    new_case,
    output_equals,
    suite,
    text,
    tool_call,
)
from smelt.report.suite import (
    SuiteResult,
    render_suite_html,
    render_suite_text,
    write_suite_report,
)


def _case(name: str, output: str, expected: str):
    return new_case(name).given(fixed_agent(output)).when(text("go")).then(output_equals(expected))


def _mixed_suite():
    return suite("my-suite").add(
        _case("ok-a", "x", "x"),
        _case("ok-b", "y", "y"),
        _case("bad-c", "actual", "expected"),
    )


# ---------------------------------------------------------------------------
# Suite builder & result model
# ---------------------------------------------------------------------------


def test_suite_builder_immutable_and_typed():
    base = suite("s")
    derived = base.add(_case("a", "x", "x"))
    assert base.cases == () and len(derived.cases) == 1
    with pytest.raises(TypeError, match="SmeltCase only"):
        base.add("not a case")


def test_suite_result_scoring():
    result = _mixed_suite().run()
    assert result.pass_count == 2
    assert not result.passed
    assert result.score == pytest.approx(2 / 3)
    assert "my-suite" in result.summary()
    with pytest.raises(AssertionError, match="suite 'my-suite'"):
        result.assert_passed()


def test_suite_all_pass_and_empty():
    assert suite("ok").add(_case("a", "x", "x")).run().passed
    empty = suite("empty").run()
    assert not empty.passed and empty.score == 0.0


# ---------------------------------------------------------------------------
# HTML overview page
# ---------------------------------------------------------------------------


def test_render_suite_html_overview():
    result = _mixed_suite().run()
    links = {r.case_name: f"{r.case_name}.html" for r in result.results}
    doc = render_suite_html(result, links)
    assert "FAIL" in doc  # 2/3 passed
    assert "2/3 passed" in doc
    assert "Score distribution" in doc
    assert 'href="ok-a.html"' in doc
    assert "1 expectation(s) failed" in doc  # note for the failing case


def test_render_suite_html_without_links_plain_text():
    result = suite("s").add(_case("solo", "x", "x")).run()
    doc = render_suite_html(result)
    assert "PASS" in doc and "solo" in doc and "href=" not in doc


def test_render_suite_html_error_note_and_escaping():
    bad = new_case("err <b>").when(text("go")).run()  # missing agent → error result
    result = SuiteResult(suite_name="s", results=[bad])
    doc = render_suite_html(result, {})
    assert "missing agent" in doc
    assert "<b>" not in doc.split("<body>")[1].split("Cases")[1]  # case name escaped


def test_score_distribution_buckets():
    results = [
        SuiteResult("s", [_case("full", "x", "x").run()]),  # 1.0
    ]
    # build a mixed result set directly
    rs = [
        _case("a", "x", "x").run(),          # 1.00
        new_case("b").when(text("g")).run(),  # 0.0 (missing agent)
    ]
    doc = render_suite_html(SuiteResult("d", rs))
    assert "<0.50" in doc and "1.00" in doc
    assert results  # silence unused


# ---------------------------------------------------------------------------
# Terminal summary
# ---------------------------------------------------------------------------


def test_render_suite_text():
    out = render_suite_text(_mixed_suite().run())
    assert "SMELT SUITE" in out and "my-suite" in out
    assert "2/3 passed" in out
    assert "✔" in out and "✘ 0.00  bad-c" in out


def test_render_suite_text_error_note():
    bad = new_case("broken").when(text("go")).run()
    out = render_suite_text(SuiteResult("s", [bad]))
    assert "missing agent" in out


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_write_suite_report_layout(tmp_path):
    result = _mixed_suite().run()
    index = write_suite_report(result, tmp_path)
    assert index.name == "index.html"
    assert index.parent.name.startswith("my-suite-")
    # per-case pages exist and are linked
    for name in ("ok-a", "ok-b", "bad-c"):
        assert (index.parent / f"{name}.html").exists()
    doc = index.read_text(encoding="utf-8")
    assert 'href="ok-a.html"' in doc
    # latest copy refreshed
    latest = tmp_path / "suites" / "my-suite-latest" / "index.html"
    assert latest.exists()


def test_write_suite_report_duplicate_case_names_disambiguated(tmp_path):
    result = suite("dup").add(_case("same", "x", "x"), _case("same", "y", "y")).run()
    index = write_suite_report(result, tmp_path)
    pages = sorted(p.name for p in index.parent.glob("same*.html"))
    assert pages == ["same-1.html", "same.html"]


def test_write_suite_report_refreshes_latest(tmp_path):
    r1 = suite("s").add(_case("first-run-case", "x", "x")).run()
    first = write_suite_report(r1, tmp_path)
    r2 = suite("s").add(_case("second-run-case", "y", "y")).run()
    write_suite_report(r2, tmp_path)
    latest_doc = (tmp_path / "suites" / "s-latest" / "index.html").read_text(encoding="utf-8")
    assert "second-run-case" in latest_doc and "first-run-case" not in latest_doc
    assert first.parent.exists()  # history kept


# ---------------------------------------------------------------------------
# Chain-level API
# ---------------------------------------------------------------------------


def test_suite_report_chain(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    result = suite("chain").add(_case("a", "x", "x")).report()
    assert result.passed
    assert result.report_path is not None
    index = tmp_path / result.report_path
    assert index.name == "index.html"
    assert "suite report written to" in capsys.readouterr().out
    assert result.assert_passed() is result


def test_suite_report_quiet_and_custom_dir(tmp_path, capsys):
    result = suite("q").add(_case("a", "x", "x")).report(output_dir=str(tmp_path / "out"), quiet=True)
    assert Path(result.report_path).parent.parent == tmp_path / "out" / "suites"
    assert capsys.readouterr().out == ""


def test_suite_report_cli(tmp_path, capsys):
    result = suite("cli").add(_case("a", "x", "x"), _case("b", "y", "z")).report_cli()
    out = capsys.readouterr().out
    assert "SMELT SUITE" in out and "1/2 passed" in out
    assert not result.passed
    quiet = suite("cli2").add(_case("a", "x", "x")).report_cli(quiet=True)
    assert quiet.passed


def test_suite_mixed_with_tool_call_case():
    case = (
        new_case("with-tool")
        .given(fixed_agent("done", tool_calls=[{"name": "run_command", "arguments": {"cmd": "ls"}}]))
        .when(text("go"))
        .then(tool_call("run_command"))
    )
    result = suite("mixed").add(case).run()
    assert result.passed
