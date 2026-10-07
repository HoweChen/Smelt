# Reference-first testing — design

Date: 2026-10-07 · Status: approved (approach B, phased; revised: references as a
first-class given fragment) · Repo: Smelt

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

## Revision (2026-10-07): references are a `given`, not an agent option

Rationale (validated against pytest-bdd's target_fixture pattern and the
skill-eval-harness manifest, where `reference` is a declared, ablatable
experiment unit): a reference file is precondition state — it belongs in
`given`. Declaring it there makes it **registered**, not just mounted: the
framework knows which files are skill references, which powers restraint
checks, coverage, and ablation. `smelt_agent(mount_resources=True)` is
dropped (subsumed).

Division of labor with `context(files=)`: context mounts **task inputs**
(files the user hands the agent); `references()` mounts **skill resources**
(with registration semantics). They coexist.

## Phase 1 — assertion layer

### 1.1 `references()` given fragment

```python
.given(references("skills/reddit"))                                # mount + register the whole skill dir's resources
.given(references("skills/reddit", only=["references/endpoints.md"]))  # precise subset / ablation
```

Implementation: a `ContextSpec` subclass/factory — mounts `references/`,
`scripts/`, `assets/` (whichever exist; `only=` filters to the listed
workspace-relative paths) into the case workspace at the same relative paths,
and records the mounted set into `CaseContext.references` (new field,
`tuple[str, ...]`, normalized posix paths). Missing skill dir → FileNotFoundError
at materialize time (consistent with context(files=)); empty resources →
registered set is empty.

### 1.2 `no_tool_call(name, args=None)`

Extend `NoToolCallExpectation` with an optional args subset (reuses
`_args_match`): a violation = a call matching name AND the args subset.
`args=None` keeps today's name-only semantics. Backward compatible.

### 1.3 Sugar assertions: `reference_read(path)` / `no_reference_read(path)`

Tool-name-agnostic: a "reference read" = any tool call carrying a string
argument that, normalized (posix, strip leading `./`, workspace-relative),
equals or ends with the given path. Rationale: skills say "read
references/x.md" without prescribing the tool (`read_file`, `read`, `cat`…).
Binary scoring, like `no_tool_call`. When the case declared `references(...)`,
both assertions also verify the path is within the declared set — reading an
undeclared file still satisfies `reference_read` (the behavior happened) but
adds a message note "not in declared references"; `no_reference_read` is
unchanged by declaration.

## Phase 2 — reference coverage report (evaluate level)

### 2.1 Declared + scanned universe

The reference universe for a skill = ∪ of every case's declared
`references(...)` set ∪ static scan of the SKILL.md body (markdown links and
inline-code spans targeting `references/`, `scripts/`, `assets/`). The scan
catches references no case declared (under-tested surface).

### 2.2 Dynamic reach

Aggregate across all behavior-case traces: the set of universe files that
appeared in any tool call (same normalization as 1.3).

### 2.3 Report

New `reference_coverage` section in `SkillEvaluation` (markdown + JSON): per
universe file → `reached` (count) or `unreached`, marking whether it came from
declaration, scan, or both; plus the unreached list. Semantics are deliberately
shallow: reach ≠ correct use. `None` (section omitted) when the universe is
empty. Evaluate without behavior cases → everything unreached, flagged "no
behavior cases ran".

### 2.4 compare() integration

`CompareResult` gains an informational `coverage_changes` list:
`reached → unreached` transitions between versions (a reference the skill
stopped consulting is a prime regression signal for reference-heavy skills).
Informational only — does not affect case verdicts or `has_regression`.

## Data flow

```
given(references(...)) ──declare──┐
                                  ├─► universe ─┐
SKILL.md ──static scan───────────┘              ├─► coverage table → report / compare
case traces ──tool calls──► reached files ──────┘
```

## Error handling

- `references()` with a missing skill dir → FileNotFoundError at materialize;
  `only=` naming nonexistent files → FileNotFoundError listing them.
- Unparseable SKILL.md links ignored (best-effort scan).
- 1.3 normalization mismatches (absolute vs relative spellings) are covered by
  tests; no silent misclassification.

## Testing

TDD throughout. Phase 1: full/partial mount, registration into CaseContext,
missing-dir and bad-only errors, args-negation pass/fail, sugar assertions
across differing tool names and path spellings (`./` prefix, absolute workspace
path), undeclared-read note. Phase 2: scan extraction (links, code spans,
non-reference links ignored), universe union (declared ∪ scanned), reach
aggregation across cases, report rendering, compare coverage_changes,
empty-universe omission.

## Non-goals

- No host-level progressive-disclosure simulation (load_skill tooling).
- No quality inference from reach counts.
- No leakage lint (answer copied from reference) in this iteration — noted as a
  follow-up, enabled by the declaration registry.
- No changes to judge prompts, weights, or grading.
