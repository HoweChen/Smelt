# Deterministic Checklist & Hybrid Suggestions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn smelt's improvement feedback into a code-first checklist — lint messages carry fix hints, suggestions are generated from lint by code (LLM only adds semantic ones), and new static checks cover the mechanical items of Anthropic's skill-authoring checklist.

**Architecture:** Extend the existing plugin-style lint framework (`Check` subclasses + `Message` dataclass). New markdown extraction helpers live in `smelt/lint/markdown.py`; checks consume them. `evaluate.py` gains `_code_suggestions` and drops lint issues from the LLM evidence.

**Tech Stack:** Python 3.14, pytest, markdown-it-py 4.2.0 (already a dependency), ruff. Project venv: `.venv/bin/python`.

Spec: `docs/superpowers/specs/2026-10-08-deterministic-checklist.md` (v2, audited).

## Global Constraints

- Run tests with `.venv/bin/python -m pytest` from the repo root `/Users/yuhaochen/Documents/codebase/projanvil/skillcheck`.
- Run lint with `.venv/bin/python -m ruff check src tests` — must stay clean.
- TDD: failing test first, watch it fail, then implement, watch it pass.
- `Message(Severity.X, "...")` positional construction must keep working (new `fix` field is keyword/third, default None).
- Fix hint texts must never contain the `|` character (rendered inside Markdown table cells).
- JSON renderers add the `"fix"` key only when not None — no new null fields.
- New lint findings are WARNING (-10 each) except metadata name-format violations (ERROR, -10 each). Message caps: heading skips 3, fence-language 3, each assets category 3.
- Commit after every task. Commit style: short imperative subject (see `git log`).

---

### Task 1: Rename examples/good_skill → good-skill

The Anthropic name charset rule (`^[a-z0-9-]+$`, added in Task 5) makes the
example fail its own lint. Rename first so every later task's full-suite run
stays green.

**Files:**
- Rename: `examples/good_skill/` → `examples/good-skill/`
- Modify: `examples/good-skill/SKILL.md` (frontmatter `name: good_skill` → `name: good-skill`)
- Modify: `tests/test_lint.py`, `tests/test_lint_extra.py`, `tests/test_evaluate.py` (path constants and name assertions)
- Modify: `README.md` (any `good_skill` mentions)

**Interfaces:**
- Consumes: nothing
- Produces: `GOOD = ROOT / "examples" / "good-skill"` in tests; `doc.name == "good-skill"`

- [ ] **Step 1: Rename directory and frontmatter**

```bash
cd /Users/yuhaochen/Documents/codebase/projanvil/skillcheck
git mv examples/good_skill examples/good-skill
```

In `examples/good-skill/SKILL.md` change the frontmatter line `name: good_skill` to `name: good-skill`. Body unchanged.

- [ ] **Step 2: Update all references**

```bash
grep -rn "good_skill" tests/ README.md src/ examples/ --include="*.py" --include="*.md"
```

Known hits (verify with the grep; update every one):
- `tests/test_lint.py`: `GOOD = EXAMPLES / "good_skill"` → `"good-skill"`; `test_load_good_skill` asserts `doc.name == "good_skill"` → `"good-skill"`; `test_discover_skills_scans_children` set `{"good_skill", ...}` → `{"good-skill", ...}`; CLI/report assertions containing `good_skill` → `good-skill`.
- `tests/test_lint_extra.py`: `GOOD = ROOT / "examples" / "good_skill"` → `"good-skill"`.
- `tests/test_evaluate.py`: same `GOOD` constant; assertions `"good_skill"` (skill_name, report title, JSON name) → `"good-skill"`.
- `tests/test_compare.py`, `tests/test_env.py`, `tests/test_repeat.py`: each defines its own `GOOD = ... / "good_skill"` constant → `"good-skill"`.
- `examples/cases/evaluate_demo.py`: two references (path and name string) → `good-skill`.
- `examples/good-skill/tests/smoke.md`: the `# good_skill smoke cases` heading → `# good-skill smoke cases`.
- `README.md`: replace `good_skill` with `good-skill` if present.
- Do NOT touch historical docs under `docs/superpowers/` or `docs/upstream-feedback-2026-10-08.md`.

- [ ] **Step 3: Run full suite to verify the rename**

Run: `.venv/bin/python -m pytest`
Expected: 323 passed (same count as before the rename).

- [ ] **Step 4: Commit**

```bash
git add -A examples tests README.md
git commit -m "examples: rename good_skill to good-skill (spec-compliant name charset)"
```

---

### Task 2: New markdown.py helpers (headings / fences / inline_code)

**Files:**
- Modify: `src/smelt/lint/markdown.py`
- Test: `tests/test_markdown.py` (new)

**Interfaces:**
- Consumes: existing `markdown_it.MarkdownIt` instance `_md` in markdown.py
- Produces (later tasks rely on these exact signatures):
  - `headings(body: str) -> list[tuple[int, str]]`
  - `fences(body: str) -> list[tuple[str, str]]` — `(info, content)`
  - `inline_code(body: str) -> list[str]`
  - `link_targets(body: str, *, include_images: bool = True) -> list[str]` (gains keyword-only param, default keeps current behavior)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_markdown.py`:

```python
"""Tests for smelt.lint.markdown tokenization helpers."""

from smelt.lint.markdown import fences, headings, inline_code, link_targets


def test_headings_ignores_hashes_inside_code_fences():
    body = "# Title\n\n```sh\n# not a heading\n## also not\n```\n\n## Real\n"
    assert headings(body) == [(1, "Title"), (2, "Real")]


def test_headings_handles_setext_and_closing_hashes():
    body = "Setext Title\n============\n\n## Closed ##\n"
    assert headings(body) == [(1, "Setext Title"), (2, "Closed")]


def test_fences_returns_info_and_content():
    body = "```python\nprint(1)\n```\n\n```\nplain\n```\n"
    assert fences(body) == [("python", "print(1)\n"), ("", "plain\n")]


def test_inline_code_collects_code_spans():
    body = "Run `scripts/a.py` then `scripts\\b.py`, not [a link](x.md)."
    assert inline_code(body) == ["scripts/a.py", "scripts\\b.py"]


def test_link_targets_can_exclude_images():
    body = "![img](assets/pic.png) and [doc](references/x.md)"
    assert link_targets(body) == ["assets/pic.png", "references/x.md"]
    assert link_targets(body, include_images=False) == ["references/x.md"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_markdown.py -v`
Expected: ImportError — `cannot import name 'fences' from 'smelt.lint.markdown'`.

- [ ] **Step 3: Implement the helpers**

In `src/smelt/lint/markdown.py`, change `link_targets` and append the helpers:

```python
def link_targets(body: str, *, include_images: bool = True) -> list[str]:
    """Resolved targets of all links (and images unless excluded), including
    reference-style links."""
    targets: list[str] = []
    for token in _md.parse(body):
        if token.type != "inline":
            continue
        for child in token.children or []:
            if child.type == "link_open":
                href = child.attrGet("href")
                if href:
                    targets.append(href)
            elif include_images and child.type == "image":
                src = child.attrGet("src")
                if src:
                    targets.append(src)
    return targets


def headings(body: str) -> list[tuple[int, str]]:
    """(level, text) per heading, from tokens — '#' lines inside code fences
    never appear here."""
    result: list[tuple[int, str]] = []
    tokens = _md.parse(body)
    for i, token in enumerate(tokens):
        if token.type == "heading_open":
            level = int(token.tag[1])  # h1..h6
            text = ""
            if i + 1 < len(tokens) and tokens[i + 1].type == "inline":
                text = tokens[i + 1].content
            result.append((level, text))
    return result


def fences(body: str) -> list[tuple[str, str]]:
    """(info, content) per fenced code block; info is the language string and
    may be empty."""
    return [(t.info, t.content) for t in _md.parse(body) if t.type == "fence"]


def inline_code(body: str) -> list[str]:
    """Contents of inline code spans (`...`)."""
    spans: list[str] = []
    for token in _md.parse(body):
        if token.type != "inline":
            continue
        for child in token.children or []:
            if child.type == "code_inline":
                spans.append(child.content)
    return spans
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_markdown.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/smelt/lint/markdown.py tests/test_markdown.py
git commit -m "lint/markdown: headings/fences/inline_code helpers, link_targets image filter"
```

---

### Task 3: Message.fix field + renderer support

**Files:**
- Modify: `src/smelt/lint/models.py` (Message dataclass)
- Modify: `src/smelt/lint/report.py` (three renderers)
- Modify: `src/smelt/evaluate.py:182` (to_dict lint message serialization)
- Test: `tests/test_lint_extra.py` (append)

**Interfaces:**
- Consumes: nothing new
- Produces: `Message(severity, text, fix=None)`; renderers append ` → {fix}` when present; JSON message dicts carry `"fix"` only when not None.

- [ ] **Step 1: Write the failing tests**

First extend the imports in `tests/test_lint_extra.py`: add `SkillReport` to
the `from smelt.lint.models import ...` line. Then append:

```python
# ---------------------------------------------------------------------------
# Message fix hints and their rendering
# ---------------------------------------------------------------------------


def test_message_fix_defaults_to_none():
    m = Message(Severity.ERROR, "broken")
    assert m.fix is None


def test_render_text_shows_fix_hint():
    report = SkillReport(
        skill_path=Path("/tmp/x"),
        results=[
            CheckResult(
                check_id="c",
                name="C",
                score=50.0,
                passed=False,
                messages=[Message(Severity.ERROR, "missing file", fix="create the file")],
            )
        ],
        total_score=50.0,
        grade="F",
    )
    from smelt.lint.report import render_text

    assert "missing file → create the file" in render_text(report)


def test_render_json_includes_fix_only_when_present():
    from smelt.lint.report import render_json

    report = SkillReport(
        skill_path=Path("/tmp/x"),
        results=[
            CheckResult(
                check_id="c",
                name="C",
                score=50.0,
                passed=False,
                messages=[
                    Message(Severity.ERROR, "a", fix="do a"),
                    Message(Severity.WARNING, "b"),
                ],
            )
        ],
        total_score=50.0,
        grade="F",
    )
    import json

    msgs = json.loads(render_json([report]))[0]["checks"][0]["messages"]
    assert msgs[0]["fix"] == "do a"
    assert "fix" not in msgs[1]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_lint_extra.py -k fix -v`
Expected: 3 failed (`Message.__init__` got an unexpected keyword / missing attributes).

- [ ] **Step 3: Implement**

`src/smelt/lint/models.py` — Message dataclass:

```python
@dataclass
class Message:
    severity: Severity
    text: str
    fix: str | None = None  # concrete remediation hint, rendered after " → "
```

`src/smelt/lint/report.py` — add a formatter and use it in all three renderers:

```python
from smelt.lint.models import Message, Severity, SkillReport

_ICON = {Severity.INFO: "ℹ", Severity.WARNING: "⚠", Severity.ERROR: "✖"}


def _fmt(m: Message) -> str:
    return f"{_ICON[m.severity]} {m.text}" + (f" → {m.fix}" if m.fix else "")
```

- `render_text`: replace `lines.append(f"        {_ICON[m.severity]} {m.text}")` with `lines.append(f"        {_fmt(m)}")`.
- `render_markdown`: replace the `msgs = ...` line with
  `msgs = "<br>".join(_fmt(m) for m in r.messages) or "—"`.
- `render_json`: replace the messages value with

```python
                    "messages": [
                        {"severity": m.severity.value, "text": m.text, **({"fix": m.fix} if m.fix else {})}
                        for m in c.messages
                    ],
```

`src/smelt/evaluate.py` line 182 — same fix rule in `to_dict`:

```python
                            "messages": [
                                {"severity": m.severity.value, "text": m.text, **({"fix": m.fix} if m.fix else {})}
                                for m in c.messages
                            ],
```

- [ ] **Step 4: Run tests to verify they pass, then full suite**

Run: `.venv/bin/python -m pytest tests/test_lint_extra.py -k fix -v && .venv/bin/python -m pytest`
Expected: 3 passed; full suite green.

- [ ] **Step 5: Commit**

```bash
git add src/smelt/lint/models.py src/smelt/lint/report.py src/smelt/evaluate.py tests/test_lint_extra.py
git commit -m "lint: Message.fix hint field, surfaced in text/markdown/json renderers"
```

---

### Task 4: Fix hints on all six existing checks

**Files:**
- Modify: `src/smelt/lint/checks/metadata.py`, `structure.py`, `trigger.py`, `clarity.py`, `assets.py`, `testcoverage.py`
- Test: `tests/test_lint.py` (append one invariant test)

**Interfaces:**
- Consumes: `Message(..., fix=...)` from Task 3
- Produces: every Message emitted by any check carries a non-None `fix`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_lint.py`:

```python
def test_every_lint_message_carries_a_fix_hint():
    for skill_dir in (BAD, EXAMPLES / "bad_ref_skill"):
        doc = load_skill(skill_dir)
        for result in run_checks(doc):
            for m in result.messages:
                assert m.fix, f"[{result.check_id}] message without fix: {m.text}"
                assert "|" not in m.fix  # would break the markdown table cell
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_lint.py::test_every_lint_message_carries_a_fix_hint -v`
Expected: FAIL — first message has `fix=None`.

- [ ] **Step 3: Add fix= to every Message construction**

`metadata.py`:
- missing frontmatter → `fix="add a --- delimited YAML frontmatter block with name and description"`
- missing name → `fix="add 'name: <skill-name>' to the frontmatter"`
- name != dir → `fix="rename the directory or the frontmatter name so they match"`
- missing description → `fix="add a description stating what the skill does and when to use it"`
- description too short → `fix="expand with what the skill does and concrete trigger scenarios"`
- description too long → `fix="trim to the essential capability and trigger scenarios"`

`structure.py`:
- only N H2 sections → `fix="split the body into sections such as When to use / Steps / Examples"`
- few sections → `fix="add an Examples or Caveats section"`
- body too thin (~N words) → `fix="add concrete steps, examples, and failure-path notes"`
- aim for IDEAL_MIN_WORDS → `fix="expand thin sections with concrete detail"`

`trigger.py`:
- no trigger guidance → `fix="state when to use the skill, e.g. 'Use when ...' or '当…时使用'"`
- weak trigger guidance → `fix="add more trigger formulations (keywords, scenarios, both languages if bilingual)"`

`clarity.py`:
- found unfilled content → `fix="replace the placeholder with real content or remove it"`
- body exceeds MAX_WORDS → existing message text already contains the advice; add `fix="move details into references/ and keep SKILL.md an overview"`

`assets.py`:
- referenced file does not exist → `fix="create the file or remove the reference"`

`testcoverage.py`:
- only inline examples → `fix="add a tests/ or evals/ directory with executable cases"`
- no test cases → `fix="add at least one behavior test or eval case"`

- [ ] **Step 4: Run test to verify it passes, then full suite**

Run: `.venv/bin/python -m pytest tests/test_lint.py::test_every_lint_message_carries_a_fix_hint -v && .venv/bin/python -m pytest`
Expected: PASS; full suite green.

- [ ] **Step 5: Commit**

```bash
git add src/smelt/lint/checks/ tests/test_lint.py
git commit -m "lint: concrete fix hints on all six checks"
```

---

### Task 5: metadata — name charset / reserved words / third-person description

**Files:**
- Modify: `src/smelt/lint/checks/metadata.py`
- Test: `tests/test_lint_extra.py` (append)

**Interfaces:**
- Consumes: `Message(..., fix=...)`
- Produces: no new public names

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_lint_extra.py`:

```python
# ---------------------------------------------------------------------------
# metadata: Anthropic frontmatter compliance
# ---------------------------------------------------------------------------


def _meta_doc(tmp_path, name="ok-name", description="Processes reports. Use when the user asks for a weekly summary."):
    # path includes the name so the existing name==dir rule does not fire
    return SkillDoc(
        path=tmp_path / name / "SKILL.md",
        name=name,
        description=description,
        frontmatter={"name": name, "description": description},
        body="body",
    )


def test_metadata_rejects_underscore_name(tmp_path):
    result = MetadataCheck().run(_meta_doc(tmp_path, name="my_skill"))
    assert any("lowercase letters, numbers and hyphens" in m.text and m.severity == Severity.ERROR for m in result.messages)


def test_metadata_rejects_long_and_reserved_names(tmp_path):
    result = MetadataCheck().run(_meta_doc(tmp_path, name="claude-" + "x" * 60))
    assert any("exceeding 64" in m.text for m in result.messages)
    assert any("reserved word" in m.text for m in result.messages)


def test_metadata_flags_first_person_description(tmp_path):
    result = MetadataCheck().run(_meta_doc(tmp_path, description="I can help you process Excel files and generate reports."))
    assert any("third person" in m.text and m.severity == Severity.WARNING for m in result.messages)


def test_metadata_accepts_compliant_frontmatter(tmp_path):
    result = MetadataCheck().run(_meta_doc(tmp_path))
    assert result.messages == []
    assert result.score == 100.0
```

(With the fixture above, `skill.dir.name` equals the frontmatter name, so the
"name == dir" rule stays silent and each test isolates its own rule.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_lint_extra.py -k metadata -v`
Expected: the three new negative/compliant tests FAIL (no such messages yet);
pre-existing metadata tests stay green.

- [ ] **Step 3: Implement**

In `metadata.py`:

```python
import re

DESC_MIN = 30
DESC_MAX = 500
NAME_MAX = 64
NAME_RE = re.compile(r"^[a-z0-9-]+$")
RESERVED_WORDS = ("anthropic", "claude")
NON_THIRD_PERSON_RE = re.compile(r"^(i['\s]|we\s|you\s)", re.IGNORECASE)
```

Replace the name block (currently `if not ... elif ... != skill.dir.name`) with:

```python
        raw_name = skill.frontmatter.get("name")
        if not raw_name:
            messages.append(
                Message(Severity.ERROR, "frontmatter is missing the name field", fix="add 'name: <skill-name>' to the frontmatter")
            )
            score -= 40
        else:
            name = str(raw_name)
            if name != skill.dir.name:
                messages.append(
                    Message(
                        Severity.WARNING,
                        f"name ({name}) does not match the directory name ({skill.dir.name})",
                        fix="rename the directory or the frontmatter name so they match",
                    )
                )
                score -= 10
            if not NAME_RE.fullmatch(name):
                messages.append(
                    Message(
                        Severity.ERROR,
                        f"name ({name}) must use lowercase letters, numbers and hyphens only",
                        fix="rename to match ^[a-z0-9-]+$, e.g. weekly-report",
                    )
                )
                score -= 10
            if len(name) > NAME_MAX:
                messages.append(
                    Message(
                        Severity.ERROR,
                        f"name is {len(name)} chars, exceeding {NAME_MAX}",
                        fix="shorten the name to 64 characters or fewer",
                    )
                )
                score -= 10
            if any(w in name.lower() for w in RESERVED_WORDS):
                messages.append(
                    Message(
                        Severity.ERROR,
                        "name contains a reserved word (anthropic/claude)",
                        fix="rename without the reserved words anthropic or claude",
                    )
                )
                score -= 10
```

Add the third-person check after the description length chain:

```python
        if desc and NON_THIRD_PERSON_RE.match(desc.strip()):
            messages.append(
                Message(
                    Severity.WARNING,
                    "description must be written in third person (no I/we/you openings)",
                    fix="rewrite in third person, e.g. 'Processes Excel files. Use when ...'",
                )
            )
            score -= 10
```

(Merge the fix= texts with Task 4's — this task's version supersedes where they overlap.)

- [ ] **Step 4: Run tests, then full suite**

Run: `.venv/bin/python -m pytest tests/test_lint_extra.py -k metadata -v && .venv/bin/python -m pytest`
Expected: new tests pass; full suite green (bad_skill gains messages but its assertions are presence-based).

- [ ] **Step 5: Commit**

```bash
git add src/smelt/lint/checks/metadata.py tests/test_lint_extra.py
git commit -m "lint/metadata: name charset and reserved-word rules, third-person description"
```

---

### Task 6: structure — heading-skip and multiple-H1 detection

**Files:**
- Modify: `src/smelt/lint/checks/structure.py`
- Test: `tests/test_lint_extra.py` (append)

**Interfaces:**
- Consumes: `headings(body)` from Task 2
- Produces: no new public names

- [ ] **Step 1: Write the failing tests**

First add `from smelt.lint.checks.structure import StructureCheck` to the
imports of `tests/test_lint_extra.py`. Then append:

```python
# ---------------------------------------------------------------------------
# structure: heading hygiene
# ---------------------------------------------------------------------------


def _struct_doc(tmp_path, body):
    # three H2s in `sections` so the existing "few sections" WARNING stays silent
    return SkillDoc(
        path=tmp_path / "SKILL.md",
        name="s",
        description="d",
        frontmatter={},
        body=body,
        sections=[(2, "A"), (2, "B"), (2, "C")],
        word_count=300,
    )


def test_structure_flags_heading_level_skip(tmp_path):
    body = "# T\n\n## A\n\n#### Deep\n\n## B\n"
    result = StructureCheck().run(_struct_doc(tmp_path, body))
    assert any("H2 to H4" in m.text for m in result.messages)


def test_structure_flags_multiple_h1(tmp_path):
    body = "# T\n\n## A\n\n# Second title\n\n## B\n"
    result = StructureCheck().run(_struct_doc(tmp_path, body))
    assert any("multiple H1" in m.text for m in result.messages)


def test_structure_ignores_hashes_in_code_fences(tmp_path):
    body = "# T\n\n## A\n\n```sh\n# comment\n## x\n```\n\n## B\n"
    result = StructureCheck().run(_struct_doc(tmp_path, body))
    assert result.messages == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_lint_extra.py -k structure -v`
Expected: `test_structure_flags_heading_level_skip` and
`test_structure_flags_multiple_h1` FAIL (no such messages yet);
`test_structure_ignores_hashes_in_code_fences` passes already — it is a
characterization guard pinning that fence contents never reach the new rules.

- [ ] **Step 3: Implement**

In `structure.py` add to `run()`, after the existing section/word checks:

```python
        heads = headings(skill.body)
        if sum(1 for lvl, _ in heads if lvl == 1) > 1:
            messages.append(
                Message(
                    Severity.WARNING,
                    "multiple H1 headings; use one H1 as the document title",
                    fix="demote extra H1 headings to H2",
                )
            )
            score -= 10
        levels = [lvl for lvl, _ in heads]
        for a, b in [(a, b) for a, b in zip(levels, levels[1:]) if b > a + 1][:3]:
            messages.append(
                Message(
                    Severity.WARNING,
                    f"heading level skips from H{a} to H{b}",
                    fix="introduce the intermediate heading level(s)",
                )
            )
            score -= 10
```

Plus `from smelt.lint.markdown import headings` at the top.

- [ ] **Step 4: Run tests, then full suite**

Run: `.venv/bin/python -m pytest tests/test_lint_extra.py -k structure -v && .venv/bin/python -m pytest`
Expected: PASS; full suite green (good-skill has one H1 and only H2 below it — no false positives).

- [ ] **Step 5: Commit**

```bash
git add src/smelt/lint/checks/structure.py tests/test_lint_extra.py
git commit -m "lint/structure: heading-skip and multiple-H1 warnings via token headings"
```

---

### Task 7: clarity — 500-line budget and unannotated code fences

**Files:**
- Modify: `src/smelt/lint/checks/clarity.py`
- Test: `tests/test_lint_extra.py` (append)

**Interfaces:**
- Consumes: `fences(body)` from Task 2
- Produces: no new public names

- [ ] **Step 1: Write the failing tests**

First add `from smelt.lint.checks.clarity import ClarityCheck` to the imports
of `tests/test_lint_extra.py`. Then append:

```python
# ---------------------------------------------------------------------------
# clarity: body line budget and fence language annotation
# ---------------------------------------------------------------------------


def test_clarity_flags_body_over_500_lines(tmp_path):
    body = "\n".join(f"line {i}" for i in range(510))
    result = ClarityCheck().run(SkillDoc(path=tmp_path / "SKILL.md", name="c", description="d", frontmatter={}, body=body))
    assert any("exceeding 500" in m.text for m in result.messages)


def test_clarity_flags_unannotated_code_like_fence(tmp_path):
    body = "Example:\n\n```\ndef f():\n    return 1\n\nclass G:\n    pass\n```\n"
    result = ClarityCheck().run(SkillDoc(path=tmp_path / "SKILL.md", name="c", description="d", frontmatter={}, body=body))
    assert any("language annotation" in m.text for m in result.messages)


def test_clarity_ignores_plain_text_fence(tmp_path):
    body = "Layout:\n\n```\nskill/\n  SKILL.md\n  scripts/\n    run.py\n```\n"
    result = ClarityCheck().run(SkillDoc(path=tmp_path / "SKILL.md", name="c", description="d", frontmatter={}, body=body))
    assert result.messages == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_lint_extra.py -k "line_budget or fence or 500 or unannotated or plain_text" -v`
Expected: the first two FAIL (no such messages); the third passes already (characterization guard).

- [ ] **Step 3: Implement**

In `clarity.py`:

```python
from smelt.lint.markdown import fences, prose_segments

MAX_WORDS = 3000
MAX_LINES = 500
# Heuristic: a fence line starting with one of these tokens is treated as
# code. Prose can still start with these words (residual false positives are
# accepted and capped at 3 messages); plain-text fences like directory trees
# do not match.
CODE_LIKE_RE = re.compile(
    r"^\s*(def |class |import \S|from \S|function |const |let |var |#!|<template>|<script>|public |private )"
)
```

In `run()`, after the word-count check:

```python
        line_count = len(skill.body.splitlines())
        if line_count > MAX_LINES:
            messages.append(
                Message(
                    Severity.WARNING,
                    f"body is {line_count} lines, exceeding {MAX_LINES}; split details into references/",
                    fix="move reference material into companion files and link them",
                )
            )
            score -= 10

        flagged = 0
        for info, content in fences(skill.body):
            if info.strip() or flagged >= 3:
                continue
            if sum(1 for ln in content.splitlines() if CODE_LIKE_RE.match(ln)) >= 2:
                flagged += 1
                messages.append(
                    Message(
                        Severity.WARNING,
                        "fenced code block without a language annotation",
                        fix="add the language after the opening fence, e.g. ```python",
                    )
                )
                score -= 10
```

- [ ] **Step 4: Run tests, then full suite**

Run: `.venv/bin/python -m pytest tests/test_lint_extra.py -v && .venv/bin/python -m pytest`
Expected: PASS; full suite green (good-skill has ~60 lines, no fences).

- [ ] **Step 5: Commit**

```bash
git add src/smelt/lint/checks/clarity.py tests/test_lint_extra.py
git commit -m "lint/clarity: 500-line body budget, unannotated code-fence warnings"
```

---

### Task 8: assets — backslash paths, anchor validity, reference depth

**Files:**
- Modify: `src/smelt/lint/checks/assets.py`
- Test: `tests/test_lint_false_positives.py` (append — same theme)

**Interfaces:**
- Consumes: `headings`, `inline_code`, `link_targets(..., include_images=False)` from Task 2
- Produces: no new public names

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_lint_false_positives.py`:

```python
# ---------------------------------------------------------------------------
# assets: path hygiene, anchors, reference depth
# ---------------------------------------------------------------------------


def test_assets_flags_backslash_paths(tmp_path):
    doc = _skill(tmp_path, "Run `scripts\\build.py` and see [doc](references\\x.md).")
    result = AssetCheck().run(doc)
    assert sum("forward slashes" in m.text for m in result.messages) == 2


def test_assets_flags_unknown_anchor(tmp_path):
    doc = _skill(
        tmp_path,
        "See [details](references/guide.md#nosuchsection).",
        files=("references/guide.md",),
    )
    (tmp_path / "references" / "guide.md").write_text("# Real Section\n")
    result = AssetCheck().run(doc)
    assert any("nosuchsection" in m.text and "anchor" in m.text for m in result.messages)


def test_assets_accepts_valid_anchor_with_github_slug_rules(tmp_path):
    (tmp_path / "references").mkdir()
    (tmp_path / "references" / "guide.md").write_text(
        "## Risks & blockers\n\n## Risks & blockers\n"
    )
    doc = _skill(tmp_path, "See [a](references/guide.md#risks--blockers) and [b](references/guide.md#risks--blockers-1).")
    result = AssetCheck().run(doc)
    assert result.messages == []


def test_assets_flags_nested_references(tmp_path):
    doc = _skill(
        tmp_path,
        "See [deep](references/deep.md).",
        files=("references/deep.md",),
    )
    (tmp_path / "references" / "deep.md").write_text("Details in [more](more.md).\n")
    result = AssetCheck().run(doc)
    assert any("one level deep" in m.text for m in result.messages)


def test_assets_anchor_check_skips_missing_files_and_non_md(tmp_path):
    doc = _skill(tmp_path, "See [gone](references/gone.md#x) and [script](scripts/s.py#main).")
    result = AssetCheck().run(doc)
    # missing-file errors fire; no anchor messages for either
    assert any("references/gone.md" in m.text for m in result.messages)
    assert not any("anchor" in m.text for m in result.messages)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_lint_false_positives.py -k "backslash or anchor or nested" -v`
Expected: 5 failed (no such messages).

- [ ] **Step 3: Implement**

In `assets.py`, extend the imports and add module-level helpers:

```python
from smelt.lint.markdown import code_contents, headings, inline_code, link_targets, prose_segments

BACKSLASH_DIR_RE = re.compile(r"^(?:scripts|templates|assets|references|examples|docs|tests)\\")
_SLUG_STRIP_RE = re.compile(r"[^\w\s-]")


def _heading_slugs(path: Path) -> set[str]:
    """GitHub-style slugs of a markdown file's headings (duplicates get -1, -2)."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return set()
    seen: dict[str, int] = {}
    slugs: set[str] = set()
    for _, title in headings(text):
        base = _SLUG_STRIP_RE.sub("", title).strip().lower().replace(" ", "-")
        n = seen.get(base, 0)
        seen[base] = n + 1
        slugs.add(base if n == 0 else f"{base}-{n}")
    return slugs
```

At the end of `AssetCheck.run()` (before `missing` computation or after — order irrelevant), add:

```python
        local_targets = [t for t in link_targets(skill.body) if not t.startswith(_EXTERNAL_PREFIXES)]

        # note: markdown-it-py URL-normalizes hrefs, so a backslash arrives as
        # %5C — unquote before testing
        backslash_hits = [t for t in local_targets if "\\" in unquote(t)]
        backslash_hits += [s for s in inline_code(skill.body) if BACKSLASH_DIR_RE.match(s)]
        for hit in backslash_hits[:3]:
            messages.append(
                Message(Severity.WARNING, f"Windows-style path with backslashes: {hit}", fix="use forward slashes in all paths")
            )
            score -= 10

        anchored = 0
        for t in local_targets:
            path_part, _, frag = t.partition("#")
            if not frag or not path_part.lower().endswith(".md"):
                continue
            target = skill.dir / unquote(urlparse(path_part).path)
            if not target.is_file():
                continue  # missing file is already reported by the existence check
            frag = unquote(frag)
            if frag not in _heading_slugs(target):
                anchored += 1
                if anchored <= 3:  # messages and deductions share the cap
                    messages.append(
                        Message(
                            Severity.WARNING,
                            f"anchor '#{frag}' not found in {path_part}",
                            fix="point the link at an existing heading or fix the anchor",
                        )
                    )
                    score -= 10

        nested = 0
        for t in local_targets:
            path_part = unquote(urlparse(t.partition("#")[0]).path)
            if not path_part.lower().endswith(".md"):
                continue
            target = skill.dir / path_part
            if not target.is_file():
                continue
            try:
                nested_links = link_targets(target.read_text(encoding="utf-8"), include_images=False)
            except OSError:
                continue
            if any(not n.startswith(_EXTERNAL_PREFIXES) for n in nested_links):
                nested += 1
                if nested <= 3:  # messages and deductions share the cap
                    messages.append(
                        Message(
                            Severity.WARNING,
                            f"{path_part} links to further local files; keep references one level deep",
                            fix="link the nested files directly from SKILL.md",
                        )
                    )
                    score -= 10
```

Note: `messages`/`score` are initialized before the missing-file block; move
the new blocks after `missing` handling but before `return self._result(...)`.

- [ ] **Step 4: Run tests, then full suite**

Run: `.venv/bin/python -m pytest tests/test_lint_false_positives.py -v && .venv/bin/python -m pytest`
Expected: all pass; full suite green. If `bad_ref_skill` gains messages, confirm no test asserts its message list (only discovery does).

- [ ] **Step 5: Commit**

```bash
git add src/smelt/lint/checks/assets.py tests/test_lint_false_positives.py
git commit -m "lint/assets: backslash paths, anchor validity, one-level-deep references"
```

---

### Task 9: Code-first suggestions in evaluate.py

**Files:**
- Modify: `src/smelt/evaluate.py` (`_build_evidence`, new `_code_suggestions`, builder `run()`, `SkillEvaluation` gains `suggestions_ran`, `to_markdown` empty state)
- Modify: `src/smelt/cli.py:69` (no-judge notice wording)
- Test: `tests/test_evaluate.py` (update `test_writing_unparseable_judge_output`; append new tests)

**Interfaces:**
- Consumes: `Message.fix` (Task 3), fix-bearing messages (Tasks 4–8)
- Produces: `_code_suggestions(lint_report, max_items) -> list[str]`; `SkillEvaluation.suggestions_ran: bool = False`

- [ ] **Step 1: Write/update the failing tests**

Update `test_writing_unparseable_judge_output` in `tests/test_evaluate.py`
(first add `from smelt.lint.models import Severity` to the test file's
imports):

```python
def test_writing_unparseable_judge_output():
    evaluation = evaluate_skill(GOOD, judge=_judge("definitely not json", SUGGESTIONS_JSON)).run()
    assert evaluation.writing is not None
    assert evaluation.writing.error is not None
    assert evaluation.writing_score is None
    # evidence is empty (good-skill is lint-clean, no behavior cases, writing
    # review failed) — the suggestions judge is never called
    assert evaluation.suggestions == []
    assert evaluation.suggestions_error is None
```

Append to `tests/test_evaluate.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_evaluate.py -k "suggestion or evidence or empty_state or not_enabled or unparseable" -v`
Expected: `test_writing_unparseable_judge_output` FAILs (suggestions still truthy today); the new tests FAIL (no `_code_suggestions` / empty-state).

- [ ] **Step 3: Implement**

In `src/smelt/evaluate.py`:

a) Imports — add `from smelt.lint.models import Severity`.

b) `SkillEvaluation` dataclass — add field after `suggestions_error`:

```python
    suggestions_ran: bool = False  # True when suggestion generation executed (even if empty)
```

c) `_build_evidence` — delete the `if evaluation.lint_report is not None:` block.

d) New function after `_build_evidence`:

```python
def _code_suggestions(lint_report: Any, max_items: int) -> list[str]:
    """Deterministic suggestions from lint messages that carry a fix hint,
    errors first (stable within severity: check order is preserved)."""
    if lint_report is None:
        return []
    ordered = [(c.check_id, m) for c in lint_report.results for m in c.messages if m.fix]
    ordered.sort(key=lambda cm: 0 if cm[1].severity is Severity.ERROR else 1)
    return [f"[{check_id}] {m.text} → {m.fix}" for check_id, m in ordered[:max_items]]
```

e) Builder `run()` — replace the suggestions block:

```python
        if self.suggestions_enabled:
            evaluation.suggestions_ran = True
            code_suggestions = _code_suggestions(evaluation.lint_report, self.suggestions_max)
            if self.judge is None:
                evaluation.suggestions = code_suggestions
                if not self.lint_enabled:
                    evaluation.suggestions_error = "no judge LLM provided; suggestion generation skipped"
            else:
                suggestions = list(code_suggestions)
                if len(suggestions) < self.suggestions_max and _build_evidence(evaluation) != "{}":
                    llm_suggestions, error = _judge_suggestions(self.judge, evaluation, self.suggestions_max)
                    seen = {s.strip() for s in suggestions}
                    for s in llm_suggestions:
                        if s.strip() not in seen:
                            suggestions.append(s)
                            seen.add(s.strip())
                    evaluation.suggestions_error = error
                evaluation.suggestions = suggestions[: self.suggestions_max]
```

f) `to_markdown` suggestions section — replace the trailing else:

```python
        lines += ["## Improvement Suggestions", ""]
        if self.suggestions:
            lines += [f"{i}. {s}" for i, s in enumerate(self.suggestions, 1)]
        elif self.suggestions_error:
            lines.append(f"Suggestion generation failed: {self.suggestions_error}")
        elif self.suggestions_ran:
            lines.append("No lint issues; nothing to suggest.")
        else:
            lines.append("(not enabled)")
```

g) `src/smelt/cli.py:69` — reword the notice:

```python
        print("ℹ no --judge-model: writing review will be marked as skipped; suggestions will be lint-derived", file=sys.stderr)
```

- [ ] **Step 4: Run tests, then full suite**

Run: `.venv/bin/python -m pytest tests/test_evaluate.py -v && .venv/bin/python -m pytest`
Expected: all pass; full suite green.

- [ ] **Step 5: Commit**

```bash
git add src/smelt/evaluate.py src/smelt/cli.py tests/test_evaluate.py
git commit -m "evaluate: code-first suggestions from lint fixes; judge only for semantic evidence"
```

---

### Task 10: Dimension convergence + docs

**Files:**
- Modify: `src/smelt/evaluate.py` (DEFAULT_DIMENSIONS)
- Modify: `README.md` (evaluate section, ~lines 316-321 and 351-353)
- Modify: `CHANGELOG.md` (new Unreleased entry)
- Test: `tests/test_evaluate.py` (append)

**Interfaces:**
- Consumes: Tasks 5–8 (static coverage) and Task 9 (suggestion pipeline)
- Produces: new 3-item DEFAULT_DIMENSIONS

- [ ] **Step 1: Write the failing test**

Append to `tests/test_evaluate.py`:

```python
def test_default_dimensions_focus_on_semantics():
    assert len(DEFAULT_DIMENSIONS) == 3
    joined = " ".join(DEFAULT_DIMENSIONS).lower()
    assert "semantic" in joined and "examples" in joined and "actionability" in joined
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_evaluate.py::test_default_dimensions_focus_on_semantics -v`
Expected: FAIL — current tuple has 5 dimensions.

- [ ] **Step 3: Implement**

In `evaluate.py`:

```python
DEFAULT_DIMENSIONS: tuple[str, ...] = (
    "Semantic accuracy (does the name/description capture what the skill actually does)",
    "Examples & edge cases (examples are correct and cover failure paths)",
    "Actionability (steps are explicit, executable, unambiguous)",
)
```

README evaluate section: change "with LLM-generated improvement suggestions
from all the evidence" to "with improvement suggestions generated code-first
from lint findings (the judge LLM only adds semantic suggestions on top)";
change the trailing paragraph to "Without a judge, the writing review is
skipped and suggestions are derived from lint findings alone, so a checklist
report is always produced."
(Editor note: the source phrase wraps across `README.md:319-321` as
"LLM-generated\nimprovement" — match across the line break.)

Also update the module docstring of `src/smelt/evaluate.py` (lines 13-16):
change "LLM improvement suggestions" to "code-first improvement suggestions
(LLM adds semantic ones)".

CHANGELOG.md — add at the top:

```markdown
## [Unreleased]

### Added

- **lint messages carry fix hints** — every finding now says how to remediate
  it, surfaced in text/markdown/json reports (` → fix` suffix, `"fix"` JSON
  key when present).
- **New static checks from Anthropic's skill-authoring checklist** —
  metadata: name charset/reserved-word rules, third-person description;
  structure: heading-skip and multiple-H1 warnings; clarity: 500-line body
  budget, unannotated code fences; assets: backslash paths, broken `#anchor`
  links, nested references (one-level-deep rule).

### Changed

- **Suggestions are code-first** — lint findings become checklist items
  without a judge; the judge LLM only adds suggestions for behavior/writing
  evidence, and is skipped entirely when there is nothing semantic to add.
  No-judge runs now always produce a suggestion list.
- **Default writing dimensions narrowed to 3 semantic ones** (semantic
  accuracy, examples & edge cases, actionability); trigger guidance and
  structure are covered statically now.
- **examples/good_skill renamed to good-skill** to comply with the name
  charset rule it now enforces.
```

- [ ] **Step 4: Run tests, full suite, and ruff**

Run: `.venv/bin/python -m pytest && .venv/bin/python -m ruff check src tests`
Expected: all green, ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/smelt/evaluate.py README.md CHANGELOG.md tests/test_evaluate.py
git commit -m "evaluate: 3 semantic default writing dimensions; docs for code-first checklist"
```

---

## Self-review notes

- Spec coverage: R1→Task 3, R2→Task 4, R3→Task 3, R4→Task 9, R5.1→Task 5,
  R5.2→Task 6, R5.3→Task 7, R5.4→Task 8, R6→Task 10, R7→Task 1, R8→Task 2,
  R9→Tasks 9 (CLI) and 10 (README/CHANGELOG).
- The acceptance criterion "judge not called when code suggestions reach
  max" is Task 9's `test_judge_skipped_when_code_suggestions_fill_max`.
- `test_disabled_parts_excluded_from_overall` and
  `test_markdown_with_error_and_disabled_parts` keep passing: lint-disabled
  + judge-less still yields the old skip message.
