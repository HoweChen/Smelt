# Reference-first testing — design

Date: 2026-10-07 · Status: approved (approach B, phased) · Repo: Smelt

## Context

Skills whose depth lives in `references/` (the minority but most complex ones —
cf. the 601-skill survey: 85.5% have no resources, but production-grade skills
like docx/pdf/pptx are reference-heavy) fail in five independent modes:

| Mode | Question | Smelt today |
|---|---|---|
| Reach | reads the right file when needed? | `tool_call(args=)` — works, manual mounting |
| Selection | picks the right one among several? | assemblable, tedious |
| Utilization | actually uses the content (not hallucinating)? | judge groundedness — works |
| Restraint | does NOT read when unneeded? | missing: args-level negation |
| Coverage | which reference is never reached by any case? | missing entirely |

Approved scope: approach B in two phases. Out of scope: full host progressive-
disclosure simulation (approach C — tests the host's routing, not skill content).

## Phase 1 — assertion layer

### 1.1 `no_tool_call(name, args=None)`

Extend `NoToolCallExpectation` with an optional args subset (reuses
`_args_match`): a violation = a call matching name AND the args subset.
`args=None` keeps today's name-only semantics. Backward compatible.

### 1.2 `smelt_agent(skill_dir, mount_resources=True)`

When the skill is a directory, copy its `references/`, `scripts/`, `assets/`
subdirectories (whichever exist) into the case workspace at the same relative
paths, before the agent loop starts. Agent-layer responsibility (it owns the
skill path); runner untouched. Default False → no behavior change.

### 1.3 Sugar assertions: `reference_read(path)` / `no_reference_read(path)`

Tool-name-agnostic: a "reference read" = any tool call carrying a string
argument that, normalized (posix, strip leading `./`, workspace-relative),
equals or ends with the given path. Rationale: skills say "read
references/x.md" without prescribing the tool (`read_file`, `read`, `cat`…).
Binary scoring, like `no_tool_call`.

## Phase 2 — reference coverage report (evaluate level)

### 2.1 Static scan

Parse the SKILL.md body for referenced paths: markdown links and inline-code
spans whose target starts with `references/`, `scripts/`, or `assets/`
(one level deep per the spec; nested paths still recorded as-is).

### 2.2 Dynamic reach

Aggregate across all behavior-case traces in the evaluation: the set of
referenced files that appeared in any tool call (same normalization as 1.3).

### 2.3 Report

New `reference_coverage` section in `SkillEvaluation` (markdown + JSON):
per referenced file → `reached` (count) or `unreached`, plus an unreached
list. Semantics are deliberately shallow: reach ≠ correct use; the report
states reach only, never infers quality. `None` when the skill references
nothing (section omitted).

### 2.4 compare() integration

`CompareResult` gains an informational `coverage_changes` list:
`reached → unreached` transitions between versions (a reference the skill
stopped consulting is a prime regression signal for reference-heavy skills).
Informational only — does not affect case verdicts or `has_regression`.

## Data flow

```
SKILL.md ──static scan──► referenced files ─┐
                                            ├─► coverage table → report / compare
case traces ──tool calls──► reached files ──┘
```

## Error handling

- Phase 1: `mount_resources` with a file (not dir) skill → ValueError at
  build time; missing subdirectories are skipped silently.
- Phase 2: unparseable SKILL.md links are ignored (scan is best-effort);
  evaluate without behavior cases → coverage is static-only
  (everything unreached, flagged as "no behavior cases ran").

## Testing

TDD throughout. Phase 1: args-negation pass/fail, mount copies + default off,
sugar assertions across differing tool names and path spellings (`./` prefix,
absolute workspace path). Phase 2: scan extraction (links, code spans,
non-reference links ignored), reach aggregation across cases, report
rendering, compare coverage_changes, empty cases.

## Non-goals

- No host-level progressive-disclosure simulation (load_skill tooling).
- No quality inference from reach counts.
- No changes to judge prompts, weights, or grading.
