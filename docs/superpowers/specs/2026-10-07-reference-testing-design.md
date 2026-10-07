# Reference-first testing — design (FINAL)

Date: 2026-10-07 · Status: approved · Repo: Smelt

## Context

Skills whose depth lives in `references/` fail in five independent modes:
reach (reads it when needed), selection (the right one), utilization (actually
uses the content), restraint (doesn't read when unneeded), coverage (no case
ever reaches a reference). Approach B, phased. Out of scope: host-level
progressive-disclosure simulation.

## Final decisions (after review rounds)

- References are a **given-layer first-class fragment**, not an agent option
  (declaration = registration; enables restraint checks, coverage, ablation).
- **No `forbid`**: the user's scenario is ablation (with/without), where the
  without arm is simply "don't mount". Discipline testing ("present but
  shouldn't read") is covered by `no_reference_read` with the file mounted.
- **Without arm = absent file, never an empty file.** Empty ≠ absent: a
  successful-but-empty read is a different, confounded world (agents mishandle
  empty results; EmpR is its own bug class). Materialized removal is the
  trusted ablation form (skill-eval-harness precedent). Cleanest arm also drops
  the pointer sentence from the skill prompt (`skill(prompt=...)` variant).
- Without-arm validity must be **proven, not assumed** — by content
  fingerprint, not just path matching (`reference_untouched`).
- `workspace_tool` confinement: deferred (escape detection via normalization
  + fingerprint suffices for v1).

## Phase 1 — API

### given

```python
.given(reference("skills/reddit/references/endpoints.md"))   # one file
.given(reference_folder("skills/reddit"))                    # whole skill's references/ + scripts/ + assets/
.given(reference_folder("skills/reddit/references"))         # or a bare directory
```

Both are factories producing `ContextSpec` (new field `references: tuple[str,
...]` — workspace-relative registered paths). `reference()`: mounts the file at
its skill-root-relative path (walk up to the dir containing SKILL.md; fallback:
basename). `reference_folder()`: skill root → mounts each existing resource
subdir under `workspace/<name>/`; bare dir → mounts recursively under
`workspace/<dir.name>/`. All mounted files are registered into
`CaseContext.references` (new field). Missing file/dir → FileNotFoundError at
materialize, naming the path.

Division of labor: `context(files=)` mounts task inputs; `reference*` mounts
skill resources with registration.

### then

```python
.then(reference_read("references/endpoints.md"))       # reach: ≥1 matching call
.then(no_reference_read("references/performance.md"))  # restraint: 0 matching calls
.then(reference_untouched("references/endpoints.md", source="skills/reddit"))  # ablation validity
.then(no_tool_call("read_file", args={"path": "..."}))  # args-level negation (extends existing)
```

Matching is tool-name-agnostic: any string argument of any tool call,
normalized (`os.path.normpath`, posix separators, `./` stripped) and compared
by equality or trailing-segment match — so absolute-path and `../` escape
attempts are detected, not missed.

`reference_untouched(path, source=...)`: proves the content never entered the
context. The evaluation layer reads the REAL file (`source/path`) and builds a
fingerprint (up to 5 content lines ≥20 chars, spread across the file); fails if
(a) any tool call targets the path, or (b) any tool RESULT contains a
fingerprint line (catches `cat`/`grep`/absolute-path leaks). A final output
containing fingerprint lines without any successful read = parametric-memory
contamination — reported in the message, still a failure for ablation validity.

### tools

```python
.given(tools(mock_tool(read_file, {"references/endpoints.md": ""})))
```

`mock_tool(base, results)` wraps a Tool: same name/spec; invoke returns the
mapped value when any string argument path-matches a key, else delegates.
For robustness testing (empty/broken/stale content) — NOT for ablation.

### Ablation pattern (no new syntax)

```python
new_case("q-with").given(reference_folder("skills/reddit")).given(agent).when(q).then(...)
new_case("q-without").given(smelt_agent(prompt=skill_body_without_pointer, ...)).when(q) \
    .then(reference_untouched("references/endpoints.md", source="skills/reddit"))
# lift = compare(with_report, without_report)
```

## Phase 2 — coverage report + compare

- Universe = ∪ of every case's registered `references` ∪ static scan of
  SKILL.md (markdown links / inline code targeting `references/`,
  `scripts/`, `assets/`).
- Reached = universe files appearing in any tool call in the aggregated
  behavior traces.
- `SkillEvaluation.reference_coverage`: per-file reached(count)/unreached +
  origin (declared/scanned); omitted when the universe is empty; markdown +
  JSON. Reach ≠ quality — stated plainly.
- `CompareResult.coverage_changes`: reached→unreached transitions between
  versions; informational only, never affects verdicts or `has_regression`.

## Error handling

- Missing file/dir at materialize → FileNotFoundError naming the path.
- `reference_untouched` with an unreadable source file → scores 0 with a
  "source not found" message (errors converge into results, consistent with
  the rest of the framework).
- Best-effort SKILL.md scan: unparseable links ignored.

## Testing

TDD throughout. Phase 1: mount/registration (full, subset, bare dir, errors),
args-negation, path normalization spellings (./, absolute, ../), fingerprint
leak detection via detour tools, contamination note, mock_tool hit/delegate.
Phase 2: scan extraction, universe union, reach aggregation, report rendering,
compare coverage_changes, empty-universe omission.

## Non-goals

- No host-level progressive-disclosure simulation; no `forbid`; no
  `workspace_tool` this iteration; no leakage lint; no quality inference from
  reach counts; no judge/weights/grading changes.
