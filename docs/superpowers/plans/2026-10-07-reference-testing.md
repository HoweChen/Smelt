# Reference-First Testing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make skill references (references/, scripts/, assets/) first-class testable objects: given-layer mounting + registration, reach/restraint/untouched assertions, content mock, and an evaluate-level coverage report with compare integration.

**Architecture:** `reference()`/`reference_folder()` are factories producing the existing `ContextSpec` (new `references` field registers mounted paths into `CaseContext.references`). Matching logic lives in one shared module `smelt/refpath.py` used by then-assertions (`smelt/then/references.py`), `mock_tool` (`smelt/tools.py`), and the evaluate coverage scan (`smelt/evaluate.py`).

**Tech Stack:** Python ≥3.11, zero new dependencies, pytest, ruff (line-length 110).

## Global Constraints

- TDD: every task writes failing tests first, watches them fail, implements minimally.
- `uv run pytest -q` must stay green; `uv run ruff check src tests examples/cases` clean; coverage gate ≥90%.
- Spec: `docs/superpowers/specs/2026-10-07-reference-testing-design.md` (FINAL). No `forbid`, no `workspace_tool`, no empty-file ablation arms.
- All new public helpers exported from `smelt/__init__.py` (keep `__all__` sorted — ruff RUF022).

---

### Task 1: `no_tool_call` args-level negation

**Files:**
- Modify: `src/smelt/then/expectations.py` (`NoToolCallExpectation`, `no_tool_call`)
- Test: `tests/test_repeat.py` is unrelated — create `tests/test_references.py` (this file will host Tasks 1–5 tests)

**Interfaces:**
- Produces: `no_tool_call(name: str, args: Mapping | None = None, *, threshold: float = 1.0)` — existing callers unaffected.

- [ ] **Step 1: failing test**

```python
# tests/test_references.py
"""Reference-testing assertions: args negation, reach, restraint, untouched, mock."""

from smelt import fixed_agent, new_case, no_tool_call, text


def test_no_tool_call_with_args_subset():
    agent = fixed_agent("done", tool_calls=[
        {"name": "read_file", "arguments": {"path": "references/endpoints.md"}},
    ])
    blocked = (
        new_case("restraint")
        .given(agent)
        .when(text("go"))
        .then(no_tool_call("read_file", args={"path": "references/endpoints.md"}))
        .run()
    )
    assert not blocked.passed
    assert "references/endpoints.md" in blocked.expectations[0].message

    other_file_ok = (
        new_case("restraint-ok")
        .given(agent)
        .when(text("go"))
        .then(no_tool_call("read_file", args={"path": "references/other.md"}))
        .run()
    )
    assert other_file_ok.passed


def test_no_tool_call_without_args_unchanged():
    agent = fixed_agent("done", tool_calls=[{"name": "read_file", "arguments": {"path": "x"}}])
    result = new_case("n").given(agent).when(text("go")).then(no_tool_call("read_file")).run()
    assert not result.passed  # name-only still blocks any call of that tool
```

- [ ] **Step 2: run, verify fail**

Run: `uv run pytest tests/test_references.py -x -q`
Expected: FAIL — `no_tool_call() got an unexpected keyword argument 'args'` (TypeError).

- [ ] **Step 3: implement**

In `src/smelt/then/expectations.py`, replace `NoToolCallExpectation` and `no_tool_call`:

```python
@dataclass(frozen=True)
class NoToolCallExpectation:
    """Assert a tool was never called (optionally: never with an argument subset).

    ``args=None`` keeps name-only semantics; with ``args`` a violation is a call
    matching the name AND the argument subset."""

    tool_name: str
    args: Mapping[str, Any] | None = None
    threshold: float = 1.0

    @property
    def name(self) -> str:
        if self.args:
            return f"no_tool_call({self.tool_name}, args={dict(self.args)})"
        return f"no_tool_call({self.tool_name})"

    def evaluate(self, trace: Trace) -> ExpectationResult:
        calls = trace.calls_named(self.tool_name)
        if self.args is not None:
            calls = [c for c in calls if _args_match(c.arguments, self.args)]
        if calls:
            msg = f"{self.tool_name} was called {len(calls)} time(s)"
            if self.args:
                msg += f" with {dict(self.args)}"
            return _result(self.name, 0.0, self.threshold, msg)
        return _result(self.name, 1.0, self.threshold)


def no_tool_call(
    name: str,
    args: Mapping[str, Any] | None = None,
    *,
    threshold: float = 1.0,
) -> NoToolCallExpectation:
    return NoToolCallExpectation(tool_name=name, args=args, threshold=threshold)
```

- [ ] **Step 4: run, verify pass**

Run: `uv run pytest tests/test_references.py -q`
Expected: 2 passed.

- [ ] **Step 5: commit**

```bash
git add src/smelt/then/expectations.py tests/test_references.py
git commit -m "no_tool_call: args-level negation"
```

---

### Task 2: `reference()` / `reference_folder()` given fragments + registration

**Files:**
- Create: `src/smelt/refpath.py` (shared normalization — Task 3 reuses it)
- Create: `src/smelt/given/references.py`
- Modify: `src/smelt/given/context.py` (ContextSpec.references field, merge, CaseContext.references, materialize)
- Modify: `src/smelt/given/__init__.py` AND `src/smelt/__init__.py` (export reference, reference_folder)
- Test: `tests/test_references.py` (append)

**Interfaces:**
- Produces: `reference(path) -> ContextSpec`, `reference_folder(directory) -> ContextSpec`;
  `ContextSpec.references: tuple[str, ...]`; `CaseContext.references: list[str]`;
  `refpath.normalize_ref_path(str) -> str`, `refpath.ref_path_matches(value, target) -> bool`,
  `refpath.calls_reading(trace, path) -> list[ToolCallRecord]`.

- [ ] **Step 1: failing tests**

```python
from pathlib import Path

import pytest

from smelt import fixed_agent, new_case, output_equals, text
from smelt.given.references import reference, reference_folder


def _skill(tmp_path) -> Path:
    d = tmp_path / "skills" / "demo"
    (d / "references").mkdir(parents=True)
    (d / "scripts").mkdir()
    (d / "SKILL.md").write_text("---\nname: demo\ndescription: x\n---\nbody", encoding="utf-8")
    (d / "references" / "a.md").write_text("alpha content line one", encoding="utf-8")
    (d / "references" / "b.md").write_text("beta content line two", encoding="utf-8")
    (d / "scripts" / "s.py").write_text("print('hi')", encoding="utf-8")
    return d


def test_reference_folder_mounts_and_registers(tmp_path):
    skill = _skill(tmp_path)
    captured = {}

    class Spy:
        def run(self, ctx, input):
            captured["refs"] = list(ctx.references)
            captured["a"] = (ctx.workspace / "references" / "a.md").read_text(encoding="utf-8")
            captured["script"] = (ctx.workspace / "scripts" / "s.py").exists()
            from smelt.trace import Trace
            return Trace(output="done")

    result = (
        new_case("mount")
        .given(reference_folder(skill))
        .given(Spy())
        .when(text("go"))
        .then(output_equals("done"))
        .run()
    )
    assert result.passed
    assert captured["a"] == "alpha content line one"
    assert captured["script"] is True
    assert sorted(captured["refs"]) == ["references/a.md", "references/b.md", "scripts/s.py"]


def test_reference_folder_bare_directory(tmp_path):
    refs = tmp_path / "refs"
    refs.mkdir()
    (refs / "x.md").write_text("x", encoding="utf-8")
    captured = {}

    class Spy:
        def run(self, ctx, input):
            captured["refs"] = list(ctx.references)
            captured["x"] = (ctx.workspace / "refs" / "x.md").exists()
            from smelt.trace import Trace
            return Trace(output="done")

    new_case("bare").given(reference_folder(refs)).given(Spy()).when(text("go")).run()
    assert captured == {"refs": ["refs/x.md"], "x": True}


def test_reference_single_file_skill_relative(tmp_path):
    skill = _skill(tmp_path)
    captured = {}

    class Spy:
        def run(self, ctx, input):
            captured["refs"] = list(ctx.references)
            captured["a"] = (ctx.workspace / "references" / "a.md").exists()
            captured["b"] = (ctx.workspace / "references" / "b.md").exists()
            from smelt.trace import Trace
            return Trace(output="done")

    new_case("single").given(reference(skill / "references" / "a.md")).given(Spy()).when(text("go")).run()
    assert captured == {"refs": ["references/a.md"], "a": True, "b": False}


def test_reference_missing_path_raises_eagerly(tmp_path):
    with pytest.raises(FileNotFoundError, match="ghost"):
        reference(tmp_path / "ghost.md")


def test_reference_folder_missing_dir_raises_eagerly(tmp_path):
    with pytest.raises(FileNotFoundError, match="ghost"):
        reference_folder(tmp_path / "ghost")


def test_stacked_references_accumulate(tmp_path):
    skill = _skill(tmp_path)
    captured = {}

    class Spy:
        def run(self, ctx, input):
            captured["refs"] = sorted(ctx.references)
            from smelt.trace import Trace
            return Trace(output="done")

    result = (
        new_case("stacked")
        .given(reference(skill / "references" / "a.md"))
        .given(reference(skill / "references" / "b.md"))
        .given(Spy())
        .when(text("go"))
        .then(output_equals("done"))
        .run()
    )
    assert result.passed
    assert captured["refs"] == ["references/a.md", "references/b.md"]  # merge accumulates


def test_reference_folder_empty_resources_registers_nothing(tmp_path):
    d = tmp_path / "skills" / "empty"
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text("---\nname: e\ndescription: x\n---\nbody", encoding="utf-8")
    captured = {}

    class Spy:
        def run(self, ctx, input):
            captured["refs"] = list(ctx.references)
            from smelt.trace import Trace
            return Trace(output="done")

    new_case("empty").given(reference_folder(d)).given(Spy()).when(text("go")).run()
    assert captured["refs"] == []
```

- [ ] **Step 2: run, verify fail**

Run: `uv run pytest tests/test_references.py -x -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'smelt.given.references'` or missing attribute.

- [ ] **Step 3: implement**

Create `src/smelt/refpath.py`:

```python
"""Workspace-relative reference path matching — shared by then-assertions,
mock_tool, and the evaluate coverage scan."""

from __future__ import annotations

import os

from smelt.trace import ToolCallRecord, Trace


def normalize_ref_path(p: str) -> str:
    """normpath + posix separators; strips leading './'. Absolute paths keep
    their tail so escape spellings still trailing-match."""
    norm = os.path.normpath(p.strip()).replace(os.sep, "/")
    while norm.startswith("./"):
        norm = norm[2:]
    return norm


def ref_path_matches(value: str, target: str) -> bool:
    t = normalize_ref_path(target)
    v = normalize_ref_path(value)
    return v == t or v.endswith("/" + t)


def calls_reading(trace: Trace, path: str) -> list[ToolCallRecord]:
    """Tool calls carrying any string argument matching the reference path."""
    hits = []
    for c in trace.tool_calls:
        if any(isinstance(v, str) and ref_path_matches(v, path) for v in c.arguments.values()):
            hits.append(c)
    return hits
```

Create `src/smelt/given/references.py`:

```python
"""given fragments for skill resources: reference() / reference_folder().

Both mount files into the case workspace AND register the workspace-relative
paths into CaseContext.references — declaration doubles as registration,
which is what restraint checks and the evaluate coverage report build on.
"""

from __future__ import annotations

import os
from pathlib import Path

from smelt.given.context import ContextSpec

RESOURCE_DIRS = ("references", "scripts", "assets")


def _skill_relative(file: Path) -> Path:
    """Path relative to the enclosing skill root (dir containing SKILL.md);
    falls back to the bare file name."""
    for parent in (file.parent, *file.parent.parents):
        if (parent / "SKILL.md").exists():
            return file.relative_to(parent)
    return Path(file.name)


def reference(path: str | os.PathLike[str]) -> ContextSpec:
    """given(reference("skills/x/references/a.md")) — mount one file at its
    skill-relative workspace path and register it."""
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"reference does not exist: {src}")
    dest = _skill_relative(src)
    return ContextSpec(files=((src, dest),), references=(dest.as_posix(),))


def reference_folder(directory: str | os.PathLike[str]) -> ContextSpec:
    """given(reference_folder("skills/x")) — mount all of the skill's resource
    dirs (references/ scripts/ assets/, whichever exist); a bare directory is
    mounted recursively under workspace/<dir.name>/. Every mounted file is
    registered."""
    d = Path(directory)
    if not d.exists():
        raise FileNotFoundError(f"reference_folder does not exist: {d}")
    entries: list[tuple[Path, Path]] = []
    if (d / "SKILL.md").exists():
        entries.extend((d / s, Path(s)) for s in RESOURCE_DIRS if (d / s).is_dir())
    else:
        entries.append((d, Path(d.name)))
    registered: list[str] = []
    for src, dst in entries:
        for f in sorted(src.rglob("*")):
            if f.is_file():
                registered.append((dst / f.relative_to(src)).as_posix())
    return ContextSpec(files=tuple(entries), references=tuple(registered))
```

In `src/smelt/given/context.py`: add `references: tuple[str, ...] = ()` to `ContextSpec`; in `merge` add `references=self.references + other.references`; in `CaseContext` add `references: list[str] = field(default_factory=list)`; in `materialize` add `self.references.extend(spec.references)`.

In `src/smelt/given/__init__.py`: export `reference` and `reference_folder` (follow existing import/export pattern there). In `src/smelt/__init__.py`: add both names to the `from smelt.given import (...)` block and to `__all__` (sorted: `reference` and `reference_folder` sit between `output_equals` and `run_case`).

- [ ] **Step 4: run, verify pass**

Run: `uv run pytest tests/test_references.py -q`
Expected: all pass (7 tests).

- [ ] **Step 5: commit**

```bash
git add src/smelt/refpath.py src/smelt/given/references.py src/smelt/given/context.py src/smelt/given/__init__.py tests/test_references.py
git commit -m "given: reference() / reference_folder() fragments with registration"
```

---

### Task 3: `reference_read` / `no_reference_read` assertions

**Files:**
- Create: `src/smelt/then/references.py`
- Modify: `src/smelt/then/__init__.py`, `src/smelt/__init__.py` (exports)
- Test: `tests/test_references.py` (append)

**Interfaces:**
- Consumes: `smelt.refpath.calls_reading`, `_result` from `smelt.then.expectations`.
- Produces: `reference_read(path, *, threshold=1.0)`, `no_reference_read(path, *, threshold=1.0)`.

- [ ] **Step 1: failing tests**

```python
from smelt import no_reference_read, reference_read


def test_reference_read_matches_any_tool_and_spelling(tmp_path):
    agent = fixed_agent("done", tool_calls=[
        {"name": "run_command", "arguments": {"cmd": "cat ./references/endpoints.md"}},
    ])
    result = (
        new_case("reach")
        .given(agent)
        .when(text("go"))
        .then(reference_read("references/endpoints.md"))
        .run()
    )
    assert result.passed  # matched via a shell command arg with ./ spelling


def test_reference_read_detects_absolute_escape(tmp_path):
    agent = fixed_agent("done", tool_calls=[
        {"name": "read_file", "arguments": {"path": "/real/disk/skills/x/references/endpoints.md"}},
    ])
    result = new_case("escape").given(agent).when(text("go")).then(reference_read("references/endpoints.md")).run()
    assert result.passed  # escape attempt is detected, not missed


def test_reference_read_fails_when_unread():
    result = (
        new_case("miss")
        .given(fixed_agent("done"))
        .when(text("go"))
        .then(reference_read("references/endpoints.md"))
        .run()
    )
    assert not result.passed


def test_no_reference_read():
    ok = new_case("r1").given(fixed_agent("done")).when(text("go")).then(no_reference_read("references/x.md")).run()
    assert ok.passed
    bad_agent = fixed_agent("done", tool_calls=[{"name": "read_file", "arguments": {"path": "references/x.md"}}])
    bad = new_case("r2").given(bad_agent).when(text("go")).then(no_reference_read("references/x.md")).run()
    assert not bad.passed
```

- [ ] **Step 2: run, verify fail**

Run: `uv run pytest tests/test_references.py -x -q`
Expected: FAIL — `ImportError: cannot import name 'reference_read'`.

- [ ] **Step 3: implement**

Create `src/smelt/then/references.py`:

```python
"""Reference assertions: reach (read when needed) and restraint (not read when
unneeded). Tool-name-agnostic: any string argument of any tool call counts."""

from __future__ import annotations

from dataclasses import dataclass

from smelt.refpath import calls_reading
from smelt.results import ExpectationResult
from smelt.then.expectations import _result
from smelt.trace import Trace


@dataclass(frozen=True)
class ReferenceReadExpectation:
    path: str
    threshold: float = 1.0

    @property
    def name(self) -> str:
        return f"reference_read({self.path})"

    def evaluate(self, trace: Trace) -> ExpectationResult:
        hits = calls_reading(trace, self.path)
        if hits:
            return _result(self.name, 1.0, self.threshold, f"read {len(hits)} time(s)")
        return _result(self.name, 0.0, self.threshold,
                       f"never read; actual calls: {trace.called_tools or '(none)'}")


@dataclass(frozen=True)
class NoReferenceReadExpectation:
    path: str
    threshold: float = 1.0

    @property
    def name(self) -> str:
        return f"no_reference_read({self.path})"

    def evaluate(self, trace: Trace) -> ExpectationResult:
        hits = calls_reading(trace, self.path)
        if hits:
            return _result(self.name, 0.0, self.threshold, f"read {len(hits)} time(s)")
        return _result(self.name, 1.0, self.threshold)


def reference_read(path: str, *, threshold: float = 1.0) -> ReferenceReadExpectation:
    """then(reference_read("references/endpoints.md")) — the reference was consulted."""
    return ReferenceReadExpectation(path=path, threshold=threshold)


def no_reference_read(path: str, *, threshold: float = 1.0) -> NoReferenceReadExpectation:
    """then(no_reference_read("references/performance.md")) — restraint."""
    return NoReferenceReadExpectation(path=path, threshold=threshold)
```

Export both (and the Expectation classes are internal) from `smelt/then/__init__.py` and `smelt/__init__.py` (keep `__all__` sorted).

- [ ] **Step 4: run, verify pass**

Run: `uv run pytest tests/test_references.py -q`
Expected: all pass.

- [ ] **Step 5: commit**

```bash
git add src/smelt/then/references.py src/smelt/then/__init__.py src/smelt/__init__.py tests/test_references.py
git commit -m "then: reference_read / no_reference_read assertions"
```

---

### Task 4: `reference_untouched` (content-fingerprint ablation validity)

**Files:**
- Modify: `src/smelt/then/references.py`
- Modify: `src/smelt/then/__init__.py`, `src/smelt/__init__.py`
- Test: `tests/test_references.py` (append)

**Interfaces:**
- Produces: `reference_untouched(path: str, *, source: str | Path, threshold=1.0)` — `source` is the skill dir (real file at `source/path`) or the file itself. Fails when: a call targets the path; a tool RESULT contains a fingerprint line; or the final output contains a fingerprint line with no successful read (contamination).

- [ ] **Step 1: failing tests**

```python
from smelt import reference_untouched

CONTENT = "The limit parameter accepts values from 1 to 100 inclusive.\nShort.\nThe after parameter is a fullname cursor for pagination."


def _skill_with_ref(tmp_path):
    d = tmp_path / "skills" / "demo"
    (d / "references").mkdir(parents=True)
    (d / "references" / "endpoints.md").write_text(CONTENT, encoding="utf-8")
    return d


def test_reference_untouched_passes_when_never_consulted(tmp_path):
    skill = _skill_with_ref(tmp_path)
    result = (
        new_case("clean")
        .given(fixed_agent("I do not know the parameters."))
        .when(text("go"))
        .then(reference_untouched("references/endpoints.md", source=skill))
        .run()
    )
    assert result.passed


def test_reference_untouched_fails_on_direct_read(tmp_path):
    skill = _skill_with_ref(tmp_path)
    agent = fixed_agent("done", tool_calls=[{"name": "read_file", "arguments": {"path": "references/endpoints.md"}}])
    result = new_case("direct").given(agent).when(text("go")).then(reference_untouched("references/endpoints.md", source=skill)).run()
    assert not result.passed


def test_reference_untouched_fails_on_detour_content_leak(tmp_path):
    # never "read" the path, but a shell command's RESULT contains the content
    skill = _skill_with_ref(tmp_path)
    agent = fixed_agent("done", tool_calls=[
        {"name": "run_command", "arguments": {"cmd": "grep -r limit ."}, "result": CONTENT},
    ])
    result = new_case("detour").given(agent).when(text("go")).then(reference_untouched("references/endpoints.md", source=skill)).run()
    assert not result.passed
    assert "leaked" in result.expectations[0].message


def test_reference_untouched_flags_parametric_memory(tmp_path):
    skill = _skill_with_ref(tmp_path)
    agent = fixed_agent("The limit parameter accepts values from 1 to 100 inclusive.")  # no tool calls at all
    result = new_case("memory").given(agent).when(text("go")).then(reference_untouched("references/endpoints.md", source=skill)).run()
    assert not result.passed
    assert "contamination" in result.expectations[0].message


def test_reference_untouched_missing_source_scores_zero(tmp_path):
    result = (
        new_case("nosource")
        .given(fixed_agent("x"))
        .when(text("go"))
        .then(reference_untouched("references/ghost.md", source=tmp_path / "nope"))
        .run()
    )
    assert not result.passed
    assert "source not found" in result.expectations[0].message
```

- [ ] **Step 2: run, verify fail**

Run: `uv run pytest tests/test_references.py -x -q`
Expected: FAIL — `ImportError: cannot import name 'reference_untouched'`.

- [ ] **Step 3: implement**

Append to `src/smelt/then/references.py`:

```python
def _fingerprint_lines(file: Path, *, max_lines: int = 5, min_len: int = 20) -> list[str]:
    """Up to max_lines distinctive content lines (>= min_len chars), spread
    across the file — the proof of 'content entered the context'."""
    lines = [ln.strip() for ln in file.read_text(encoding="utf-8").splitlines()]
    lines = [ln for ln in lines if len(ln) >= min_len]
    if len(lines) <= max_lines:
        return lines
    step = len(lines) / max_lines
    return [lines[int(i * step)] for i in range(max_lines)]


@dataclass(frozen=True)
class ReferenceUntouchedExpectation:
    """Ablation validity: the reference's content never entered the context.

    Checks calls (path match), then tool RESULTS (fingerprint leak via detour
    tools), then the final output (parametric-memory contamination)."""

    path: str
    source: Path
    threshold: float = 1.0

    @property
    def name(self) -> str:
        return f"reference_untouched({self.path})"

    def evaluate(self, trace: Trace) -> ExpectationResult:
        calls = calls_reading(trace, self.path)
        if calls:
            return _result(self.name, 0.0, self.threshold,
                           f"read via {calls[0].name}({calls[0].arguments})")
        real = Path(self.source)
        if real.is_dir():
            real = real / self.path
        if not real.exists():
            return _result(self.name, 0.0, self.threshold, f"source not found: {real}")
        fingerprint = _fingerprint_lines(real)
        for c in trace.tool_calls:
            payload = "" if c.result is None else json.dumps(c.result, ensure_ascii=False, default=str)
            for line in fingerprint:
                if line in payload:
                    return _result(self.name, 0.0, self.threshold,
                                   f"content leaked into result of {c.name}: {line[:60]!r}")
        for line in fingerprint:
            if line in trace.output:
                return _result(
                    self.name, 0.0, self.threshold,
                    "output contains reference content with no read — "
                    f"parametric-memory contamination: {line[:60]!r}",
                )
        return _result(self.name, 1.0, self.threshold, "untouched")


def reference_untouched(path: str, *, source: str | Path, threshold: float = 1.0) -> ReferenceUntouchedExpectation:
    """then(reference_untouched("references/endpoints.md", source="skills/reddit"))."""
    return ReferenceUntouchedExpectation(path=path, source=Path(source), threshold=threshold)
```

Add `import json` and `from pathlib import Path` at the top of `src/smelt/then/references.py`. Export `reference_untouched` from `smelt/then/__init__.py` and `smelt/__init__.py`.

- [ ] **Step 4: run, verify pass**

Run: `uv run pytest tests/test_references.py -q`
Expected: all pass.

- [ ] **Step 5: commit**

```bash
git add src/smelt/then/references.py src/smelt/then/__init__.py src/smelt/__init__.py tests/test_references.py
git commit -m "then: reference_untouched with content-fingerprint leak detection"
```

---

### Task 5: `mock_tool` (content-level ablation / robustness)

**Files:**
- Modify: `src/smelt/tools.py`
- Modify: `src/smelt/__init__.py`
- Test: `tests/test_references.py` (append)

**Interfaces:**
- Consumes: `Tool` from `smelt.tools`, `ref_path_matches` from `smelt.refpath`.
- Produces: `mock_tool(base: Tool, results: Mapping[str, Any]) -> Tool`.

- [ ] **Step 1: failing test**

```python
from pathlib import Path

from smelt import mock_tool, tool


def test_mock_tool_intercepts_matching_path(tmp_path):
    real = tmp_path / "real.md"
    real.write_text("real content", encoding="utf-8")

    @tool
    def read_file(path: str) -> str:
        """Read a file"""
        return Path(path).read_text(encoding="utf-8")

    mocked = mock_tool(read_file, {"references/endpoints.md": ""})
    assert mocked.name == "read_file"
    assert mocked.spec()["parameters"] == read_file.spec()["parameters"]
    assert mocked.invoke({"path": "references/endpoints.md"}) == ""
    assert mocked.invoke({"path": str(real)}) == "real content"  # non-matching delegates
```

- [ ] **Step 2: run, verify fail**

Run: `uv run pytest tests/test_references.py -x -q`
Expected: FAIL — `ImportError: cannot import name 'mock_tool'`.

- [ ] **Step 3: implement**

Append to `src/smelt/tools.py`:

```python
def mock_tool(base: Tool, results: Mapping[str, Any]) -> Tool:
    """Wrap a tool: when any string argument path-matches a key in ``results``,
    return the mapped value instead of calling the real handler. For
    content-level robustness tests (empty / broken / stale reference content) —
    NOT for with/without ablation (there, simply don't mount)."""
    from smelt.refpath import ref_path_matches

    def handler(**kwargs: Any) -> Any:
        for key, value in results.items():
            if any(isinstance(v, str) and ref_path_matches(v, key) for v in kwargs.values()):
                return value
        return base.handler(**kwargs)

    return Tool(name=base.name, description=base.description, handler=handler,
                parameters=dict(base.parameters))
```

Export `mock_tool` from `smelt/__init__.py`.

- [ ] **Step 4: run, verify pass**

Run: `uv run pytest tests/test_references.py -q`
Expected: all pass.

- [ ] **Step 5: commit**

```bash
git add src/smelt/tools.py src/smelt/__init__.py tests/test_references.py
git commit -m "tools: mock_tool for content-level robustness tests"
```

---

### Task 6: evaluate reference coverage + compare integration

**Files:**
- Modify: `src/smelt/runner.py` (CaseResult carries registered references)
- Modify: `src/smelt/results.py` (`CaseResult.references` field)
- Modify: `src/smelt/evaluate.py` (scan + coverage computation + report/JSON)
- Modify: `src/smelt/compare.py` (`coverage_changes`)
- Test: `tests/test_references.py` (append)

**Interfaces:**
- `CaseResult.references: list[str]` (registered refs, from the last successful run's context).
- `SkillEvaluation.reference_coverage: list[dict] | None` — entries `{path, reached: int, origin: "declared"|"scanned"|"both"}`; in `to_dict()` and a `## Reference Coverage` markdown section.
- `CompareResult.coverage_changes: list[str]` — `"<path>: reached → unreached"` entries; informational.

- [ ] **Step 1: failing tests**

```python
from smelt import evaluate_skill


def test_evaluate_reference_coverage(tmp_path):
    skill = _skill(tmp_path)  # has references/a.md, references/b.md
    (skill / "SKILL.md").write_text(
        "---\nname: demo\ndescription: x\n---\nRead [a](references/a.md) when needed; also see `references/missing-from-cases.md`.",
        encoding="utf-8",
    )
    reader = fixed_agent("done", tool_calls=[{"name": "read_file", "arguments": {"path": "references/a.md"}}])
    case = (
        new_case("uses-a")
        .given(reference_folder(skill))
        .given(reader)
        .when(text("go"))
        .then(reference_read("references/a.md"))
    )
    evaluation = (
        evaluate_skill(skill)
        .with_cases(case)
        .with_lint(False)
        .with_writing(enabled=False)
        .with_suggestions(enabled=False)
        .with_times(1)
        .run()
    )
    coverage = {c["path"]: c for c in evaluation.reference_coverage}
    assert coverage["references/a.md"]["reached"] == 1 and coverage["references/a.md"]["origin"] == "both"
    assert coverage["references/b.md"]["reached"] == 0
    assert coverage["references/missing-from-cases.md"]["origin"] == "scanned"
    md = evaluation.to_markdown()
    assert "## Reference Coverage" in md and "unreached" in md


def test_evaluate_coverage_omitted_without_references(tmp_path):
    # a skill whose SKILL.md mentions no references/ scripts/ assets/ paths
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "SKILL.md").write_text(
        "---\nname: plain\ndescription: x\n---\nNo resources here.", encoding="utf-8"
    )
    case = new_case("c").given(fixed_agent("done")).when(text("go")).then(output_equals("done"))
    evaluation = (
        evaluate_skill(plain).with_cases(case)
        .with_lint(False).with_writing(enabled=False).with_suggestions(enabled=False).with_times(1).run()
    )
    assert evaluation.reference_coverage is None
    assert "Reference Coverage" not in evaluation.to_markdown()


def test_compare_coverage_changes(tmp_path):
    base_payload = {
        "skill": {"name": "v1"}, "overall": {"score": 80.0}, "behavior": [],
        "reference_coverage": [{"path": "references/a.md", "reached": 1, "origin": "both"}],
    }
    cand_payload = {
        "skill": {"name": "v2"}, "overall": {"score": 80.0}, "behavior": [],
        "reference_coverage": [{"path": "references/a.md", "reached": 0, "origin": "both"}],
    }
    from smelt import compare

    result = compare(base_payload, cand_payload)
    assert result.coverage_changes == ["references/a.md: reached → unreached"]
    assert not result.has_regression  # informational only
    assert "reached → unreached" in result.to_markdown()  # rendered for humans


def test_references_survive_repeated_aggregation(tmp_path):
    skill = _skill(tmp_path)
    agent = fixed_agent("done", tool_calls=[{"name": "read_file", "arguments": {"path": "references/a.md"}}])
    case = (
        new_case("rep")
        .given(reference_folder(skill))
        .given(agent)
        .when(text("go"))
        .then(reference_read("references/a.md"))
    )
    evaluation = (
        evaluate_skill(skill).with_cases(case)
        .with_lint(False).with_writing(enabled=False).with_suggestions(enabled=False)
        .with_times(3).run()
    )
    # aggregation keeps the last ok run's registered references
    assert evaluation.behavior_results[0].references  # non-empty
    coverage = {c["path"]: c for c in evaluation.reference_coverage}
    assert coverage["references/a.md"]["reached"] >= 1
```

- [ ] **Step 2: run, verify fail**

Run: `uv run pytest tests/test_references.py -x -q`
Expected: FAIL — `AttributeError` / assertion errors (no reference_coverage).

- [ ] **Step 3: implement**

`src/smelt/results.py` — add to `CaseResult`:

```python
    references: list[str] = field(default_factory=list)  # registered reference paths (from context)
```

`src/smelt/runner.py` — in `_run_once`, after `ctx.materialize(...)` and a successful run, pass `references=list(ctx.references)` into the returned `CaseResult`; in `_aggregate`'s final return, add `references=last.references`.

`src/smelt/evaluate.py` — add:

```python
_REF_SCAN_RE = re.compile(r"(?:\]\(|`)((?:references|scripts|assets)/[^)`\s]+)")


def _scan_skill_refs(document: str) -> list[str]:
    """Referenced resource paths in markdown links / inline code spans."""
    return sorted(set(_REF_SCAN_RE.findall(document)))
```

Add to `SkillEvaluation`: field `reference_coverage: list[dict[str, Any]] | None = None`. In `SkillEvaluationBuilder.run()`, after behavior results and lint, compute (requires `import re` at top of evaluate.py):

```python
        declared = sorted({r for res in evaluation.behavior_results for r in res.references})
        scanned = _scan_skill_refs(self._read_document())
        universe = sorted(set(declared) | set(scanned))
        if universe:
            from smelt.refpath import calls_reading

            evaluation.reference_coverage = [
                {
                    "path": p,
                    "reached": sum(
                        len(calls_reading(res.trace, p))
                        for res in evaluation.behavior_results
                        if res.trace is not None
                    ),
                    "origin": ("both" if p in declared and p in scanned
                               else "declared" if p in declared else "scanned"),
                }
                for p in universe
            ]
```

In `to_dict()`: add `"reference_coverage": self.reference_coverage`. In `to_markdown()`, after the Behavior Tests section:

```python
        if self.reference_coverage:
            lines += ["## Reference Coverage", "", "| Reference | Reached | Origin |", "|---|---|---|"]
            for c in self.reference_coverage:
                reached = f"{c['reached']}×" if c["reached"] else "unreached"
                lines.append(f"| {c['path']} | {reached} | {c['origin']} |")
            lines.append("")
```

`src/smelt/compare.py` — add to `CompareResult`:

```python
    coverage_changes: list[str] = field(default_factory=list)  # informational reached→unreached transitions
```

and in `compare()`, before the return:

```python
    def _reached(payload):
        return {c["path"]: c.get("reached", 0) > 0 for c in (payload.get("reference_coverage") or [])}

    base_reach, cand_reach = _reached(base), _reached(cand)
    changes = [
        f"{p}: reached → unreached"
        for p in base_reach
        if base_reach[p] and not cand_reach.get(p, False)
    ]
```

then pass `coverage_changes=changes` to the `CompareResult(...)` constructor. Also render it: in `to_dict()` add `"coverage_changes": self.coverage_changes`, and at the end of `to_markdown()` (before the regressions block) add:

```python
        if self.coverage_changes:
            lines.append("**Reference coverage changes:** " + "; ".join(self.coverage_changes))
            lines.append("")
```

- [ ] **Step 4: run, verify pass**

Run: `uv run pytest tests/test_references.py -q && uv run pytest -q`
Expected: all pass (full suite green).

- [ ] **Step 5: commit**

```bash
git add src/smelt/results.py src/smelt/runner.py src/smelt/evaluate.py src/smelt/compare.py tests/test_references.py
git commit -m "evaluate: reference coverage report; compare: coverage_changes"
```

---

### Task 7: docs, example, changelog

**Files:**
- Modify: `README.md` (given section row + a "Testing references" subsection under then)
- Create: `examples/cases/reference_cases.py`
- Modify: `CHANGELOG.md` (new Unreleased/0.4.0 section)

- [ ] **Step 1: example**

`examples/cases/reference_cases.py` — full content:

```python
"""Reference testing demo: reach, restraint, ablation validity.

Usage: uv run smelt run examples/cases/reference_cases.py -v
All three cases pass (exit 0).
"""

import tempfile
from pathlib import Path

from smelt import (
    LLMResponse,
    ScriptedLLM,
    new_case,
    no_reference_read,
    output_contains,
    reference_folder,
    reference_read,
    reference_untouched,
    smelt_agent,
    text,
    tool,
)

# a demo skill whose depth lives in references/
SKILL_DIR = Path(tempfile.mkdtemp(prefix="smelt-ref-demo-"))
(SKILL_DIR / "references").mkdir()
(SKILL_DIR / "SKILL.md").write_text(
    """---
name: reddit
description: Query Reddit endpoints
---

# Reddit Skill

For exact parameter lists, read references/endpoints.md first — do not guess.
""",
    encoding="utf-8",
)
(SKILL_DIR / "references" / "endpoints.md").write_text(
    "# Endpoints\n- GET /r/{sub}/new — params: limit (1-100), after cursor\n",
    encoding="utf-8",
)


@tool
def read_file(path: str) -> str:
    """Read a file"""
    return Path(path).read_text(encoding="utf-8")


# 1) reach: the task needs exact params → the agent must consult the reference
reach = (
    new_case("ref-reach")
    .given(reference_folder(SKILL_DIR))
    .given(smelt_agent(
        SKILL_DIR,
        llm=ScriptedLLM([
            LLMResponse.call("read_file", {"path": "references/endpoints.md"}),
            LLMResponse.say("limit (1-100) and after cursor"),
        ]),
        tools=[read_file],
    ))
    .when(text("exact params of /new?"))
    .then(reference_read("references/endpoints.md"))
    .then(output_contains("limit"))
)

# 2) restraint: common question → the reference stays unread
restraint = (
    new_case("ref-restraint")
    .given(reference_folder(SKILL_DIR))
    .given(smelt_agent(
        SKILL_DIR,
        llm=ScriptedLLM([LLMResponse.say("/hot lists current hot posts")]),
        tools=[read_file],
    ))
    .when(text("what does /hot do?"))
    .then(no_reference_read("references/endpoints.md"))
)

# 3) ablation validity (without arm): nothing mounted, and the fingerprint
#    proves the content never entered the context
without = (
    new_case("ref-ablation-without")
    .given(smelt_agent(
        llm=ScriptedLLM([LLMResponse.say("I do not have the parameter list.")]),
        system_prompt="You answer Reddit API questions.",
        tools=[read_file],
    ))
    .when(text("exact params of /new?"))
    .then(reference_untouched("references/endpoints.md", source=SKILL_DIR))
)

cases = [reach, restraint, without]
```

Run: `uv run smelt run examples/cases/reference_cases.py -v`
Expected: `3/3 cases passed`.

- [ ] **Step 2: README** — add to the given table:

```markdown
| `reference(path)` / `reference_folder(dir)` | mount skill resources (references/, scripts/, assets/) into the workspace and register them for coverage |
```

then rows: `reference_read` / `no_reference_read` / `reference_untouched(path, source=...)`, and a short "Testing references & ablation" subsection with the with/without pattern (mount vs omit + compare) and the rule: never use empty files for without arms.

- [ ] **Step 3: CHANGELOG** — 0.4.0 section listing the five additions.

- [ ] **Step 4: verify + commit**

Run: `uv run smelt run examples/cases/reference_cases.py -v` → all pass; `uv run pytest -q` green; `uv run ruff check src tests examples/cases` clean.

```bash
git add README.md CHANGELOG.md examples/cases/reference_cases.py
git commit -m "docs: reference testing guide + example; changelog 0.4.0"
```
