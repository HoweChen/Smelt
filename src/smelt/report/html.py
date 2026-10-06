"""HTML report rendering for a single CaseResult — self-contained, inline CSS."""

from __future__ import annotations

import html
import json
from datetime import UTC, datetime

from smelt.results import CaseResult


def _esc(value: object) -> str:
    return html.escape(str(value))


BASE_CSS = """
  :root { --pass:#16a34a; --fail:#dc2626; --bg:#0f172a; --card:#1e293b; --fg:#e2e8f0; --muted:#94a3b8; }
  * { box-sizing:border-box; }
  body { font-family:-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; background:var(--bg);
         color:var(--fg); margin:0; padding:2rem; line-height:1.5; }
  .container { max-width:960px; margin:0 auto; }
  header { display:flex; align-items:center; gap:1rem; margin-bottom:1.5rem; }
  h1 { font-size:1.4rem; margin:0; }
  h2 { font-size:1.05rem; color:var(--muted); text-transform:uppercase; letter-spacing:.05em; }
  .badge { padding:.25rem .9rem; border-radius:9999px; font-weight:700; color:#fff; }
  .badge.PASS { background:var(--pass); } .badge.FAIL { background:var(--fail); }
  .meta { color:var(--muted); font-size:.85rem; margin-bottom:1.5rem; }
  .meta code { color:var(--fg); }
  section { background:var(--card); border-radius:.6rem; padding:1.1rem 1.3rem; margin-bottom:1.2rem; }
  table { width:100%; border-collapse:collapse; font-size:.92rem; }
  th, td { text-align:left; padding:.45rem .6rem; border-bottom:1px solid #334155; vertical-align:top; }
  th { color:var(--muted); font-weight:600; }
  .bar { display:inline-block; width:140px; height:8px; background:#334155; border-radius:4px;
         overflow:hidden; margin-right:.5rem; vertical-align:middle; }
  .bar-fill { height:100%; } .bar-fill.pass { background:var(--pass); } .bar-fill.fail { background:var(--fail); }
  .pass { color:var(--pass); } .fail { color:var(--fail); }
  .num { font-variant-numeric:tabular-nums; }
  .msg { color:var(--muted); font-size:.85rem; }
  pre { background:#0b1220; padding:.8rem 1rem; border-radius:.4rem; overflow-x:auto;
         white-space:pre-wrap; word-break:break-word; }
  pre.error { border-left:3px solid var(--fail); }
  .muted { color:var(--muted); }
  footer { color:var(--muted); font-size:.8rem; text-align:center; margin-top:2rem; }
  a { color:#60a5fa; text-decoration:none; } a:hover { text-decoration:underline; }
"""


def _score_class(score: float, threshold: float) -> str:
    return "pass" if score >= threshold else "fail"


def render_html(result: CaseResult) -> str:
    """Render one CaseResult as a self-contained HTML page."""
    status = "PASS" if result.passed else "FAIL"
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

    rows = []
    for e in result.expectations:
        pct = round(e.score * 100, 1)
        cls = _score_class(e.score, e.threshold)
        rows.append(f"""
      <tr>
        <td><code>{_esc(e.name)}</code></td>
        <td>
          <div class="bar"><div class="bar-fill {cls}" style="width:{pct}%"></div></div>
          <span class="num">{e.score:.2f}</span> / threshold {e.threshold:.2f}
        </td>
        <td class="{cls}">{'✔' if e.passed else '✘'}</td>
        <td class="msg">{_esc(e.message)}</td>
      </tr>""")
    expectations_table = (
        f"""<table>
    <thead><tr><th>Expectation</th><th>Score</th><th></th><th>Notes</th></tr></thead>
    <tbody>{''.join(rows)}
    </tbody>
  </table>"""
        if rows
        else "<p class='muted'>No expectations (flow-only run).</p>"
    )

    error_block = (
        f"<section><h2>Error</h2><pre class='error'>{_esc(result.error)}</pre></section>"
        if result.error
        else ""
    )

    trace_section = ""
    if result.trace is not None:
        calls = "".join(
            f"<tr><td><code>{_esc(c.name)}</code></td>"
            f"<td><code>{_esc(json.dumps(c.arguments, ensure_ascii=False))}</code></td>"
            f"<td class='{'fail' if c.error else 'pass'}'>{_esc(c.error) if c.error else 'ok'}</td></tr>"
            for c in result.trace.tool_calls
        )
        calls_table = (
            f"""<table>
      <thead><tr><th>Tool</th><th>Arguments</th><th>Result</th></tr></thead>
      <tbody>{calls}</tbody>
    </table>"""
            if calls
            else "<p class='muted'>No tool calls.</p>"
        )
        trace_section = f"""
  <section>
    <h2>Final output</h2>
    <pre>{_esc(result.trace.output)}</pre>
  </section>
  <section>
    <h2>Tool calls ({len(result.trace.tool_calls)})</h2>
    {calls_table}
  </section>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Smelt report — {_esc(result.case_name)}</title>
<style>{BASE_CSS}</style>
</head>
<body>
<div class="container">
  <header>
    <span class="badge {status}">{status}</span>
    <h1>{_esc(result.case_name)}</h1>
    <span>score {result.score:.2f}</span>
  </header>
  <div class="meta">
    workspace <code>{_esc(result.workspace or '-')}</code> · generated {generated}
  </div>
  {error_block}
  <section>
    <h2>Expectations ({len(result.expectations)})</h2>
    {expectations_table}
  </section>
  {trace_section}
  <footer>generated by smelt</footer>
</div>
</body>
</html>
"""
