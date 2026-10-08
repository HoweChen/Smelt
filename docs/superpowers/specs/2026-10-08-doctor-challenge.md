# Spec: smelt doctor + evaluate challenge (adversarial critique loop)

Date: 2026-10-08
Status: revised after audit (v2; merged from docs/smelt-doctor-rfc.md + docs/smelt-challenge-rfc.md)

## Background

Smelt cases are mostly written by an agent. AI-written cases have a systematic
risk: they are easy and loosely asserted, so they pass whether the skill is
good or bad ("false green"). And nothing probes the skill adversarially —
nobody tries inputs that should trigger but are phrased oddly, should NOT
trigger but look similar, or fixtures with distracting content.

This spec adds **critique as a first-class mechanism**, split by what is being
questioned:

- `smelt doctor` questions the **yardstick**: are the cases sensitive
  (mutation check)? Is the judge calibrated (judge canary)?
- `evaluate`'s new **challenge** part questions the **skill**: a challenger
  agent generates adversarial probes that run against the skill live.

Design principles (from the RFCs):

1. LLM critique proposes; deterministic checks dispose. Kill verdicts reuse
   `compare()`'s noise band; canary uses `fixed_agent` (fully deterministic).
2. `doctor` runs on the *case* iteration cadence; `challenge` runs on the
   *skill* iteration cadence (inside every `evaluate`).
3. Challenge results never enter `overall_score` by default — the weighted
   score is the anchor for `compare()` and `--fail-under` and must stay stable.
4. Missing doctor agent is a hard error (exit 2) — doctor's entire job is
   verification, silently skipping is false green. Missing challenger in
   evaluate degrades gracefully (consistent with writing/suggestions).

## Goal

1. Role-based LLM configuration: each role (JUDGE / AGENT / DOCTOR /
   CHALLENGER) gets an independent provider/base_url/api_key/model quartet.
2. `smelt doctor`: mutation check over a case file + judge canary, with a
   health report and CI gate.
3. `evaluate_skill(...).with_challenge()`: adversarial probes on by default,
   reported as an advisory section.
4. `@mutate_check(guards=...)`: per-case guard claims verified by doctor.

## Non-goals (out of scope)

- Auto-writing probed/surviving-mutant cases into case files (v1 reports
  suggested case code only; human review promotes).
- Custom canary traces (v1 ships one built-in bad trace).
- Challenge score in the default `overall_score` weights.
- Multi-challenger debate; semantic dedup of probes vs existing cases.
- Changing existing scoring, weights, or `then/judge.py` behavior.

## Requirements

### R1. Role-based LLM configuration (`smelt/llm_config.py`, new)

New `LLMConfig` dataclass with fields `provider` ("openai" | "anthropic",
default "openai"), `base_url: str | None`, `api_key: str | None`,
`model: str | None`.

- `LLMConfig.from_role(role, *, provider=None, base_url=None, api_key=None,
  model=None)` — explicit kwargs win; then `SMELT_<ROLE>_<KEY>`; then the
  legacy shared keys `SMELT_<KEY>` (`SMELT_API_KEY`, `SMELT_BASE_URL`,
  `SMELT_LLM_PROVIDER`) as common fallback. ROLE is uppercased; valid roles:
  `judge`, `agent`, `doctor`, `challenger`.
- `.build()` returns an `LLMClient`: reuses the existing provider-factory
  logic (langchain extras) for provider="anthropic"; plain
  `OpenAIChatClient` for provider="openai" with explicit base_url/api_key.
- `.build()` raises `LLMConfigError` (new) when **model** is unresolvable —
  model is the only hard requirement. `api_key`/`base_url` may be None: the
  underlying client keeps its own env fallback (e.g. `OPENAI_API_KEY`),
  matching what `OpenAIChatClient` tolerates today.
- `smelt/env.py` docstring updated; legacy keys keep working unchanged.
- Env auto-load (`_auto_load`) fires inside `from_role`, same as
  `OpenAIChatClient` today.

### R2. `smelt doctor` CLI + `doctor()` Python API

- New module `src/smelt/challenge/` with `__init__.py`, `mutation.py`,
  `canary.py`, `doctor.py`.
- `doctor(case_files, *, skill, doctor: LLMClient | None = None,
  min_score: float = 0.8) -> DoctorReport` (builder not needed; one call).
- CLI: `smelt doctor <cases.py...> --skill <path> [--doctor-provider P]
  [--doctor-model M] [--doctor-base-url U] [--doctor-api-key K]
  [--min-score 0.8] [--output path]`.
- Doctor agent resolution: `--doctor-*` flags > `SMELT_DOCTOR_*` env >
  legacy shared keys. If no doctor model can be resolved: print an error
  naming the missing keys and exit 2 (hard error, no silent skip).
- Exit codes: 0 healthy; 1 health issue found (mutation score below
  `--min-score`, surviving mutants reported, false guards, dangling guards,
  or canary miscalibrated); 2 configuration/usage error.

### R3. Mutation operators (`smelt/challenge/mutation.py`)

Deterministic transforms on an isolated copy of the skill (original
directory never touched; copy under a temp dir per run):

| operator | action |
|---|---|
| `drop_section:<heading>` | remove one `##` section (body included) from the copied SKILL.md |
| `drop_reference:<path>` | delete the reference file from the copied skill directory |
| `drop_constraint` | remove prohibition sentences ("do not / never / 不要") |
| `drop_requirement` | remove mandate sentences ("must / always / 必须") |

v1 operator set: `drop_section` (one mutant per `##` section),
`drop_reference` (one per declared/scanned reference), `drop_constraint`,
`drop_requirement` (one mutant each, all matching lines removed).
`weaken_trigger` / `invert_condition` from the RFC are deferred — sentence
inversion proved too unreliable for v1.

Sentence-level operators use a conservative regex line filter; a mutant that
produces an unparseable SKILL.md is skipped and reported as `invalid`, not
counted in the score.

### R4. Mutation run + kill verdict

- **Baseline**: participating cases run once against the unmutated skill
  copy with evaluate's normal repeat policy (`with_times` > per-case
  `.repeat()` > auto). The baseline supplies both the reference scores and
  σ (`score_std`) for the kill band.
- **Mutant runs**: each mutant runs every participating case with `times=1`
  (forced, overriding repeats) against the mutated skill copy. Agent binding
  follows evaluate's rules: cases with their own agent keep it; unbound
  cases bind via `smelt_agent(mutated_skill_copy, llm=doctor_agent,
  tools=...)`.
- Kill verdict per (mutant, case): mutant case score < baseline case score
  by more than `max(0.05, 2σ)` where σ is the baseline's `score_std` (σ=0
  when baseline runs=1 — then the 0.05 floor decides). Same formula as
  `compare()`.
- Mutant verdict: `killed` if ANY case kills it (suite level, short-circuit
  allowed); `survived` otherwise. `mutation_score = killed / (killed +
  survived)`; invalid mutants excluded.
- Cases whose agent is deterministic (reuse `_is_deterministic` from
  evaluate.py — move it to a shared spot or import) do not participate:
  when no participating case remains, report mutation as
  `N/A (deterministic backend)` instead of a misleading 0%.

### R5. `@mutate_check` decorator (`smelt/challenge/mutation.py`, exported)

```python
@mutate_check(guards="section:使用边界")
def merge_commit_refusal(): ...
```

- Module-level registry: the decorator records `(func, MutateSpec)` at
  import time.
- **Loading**: `_load_cases` (cli.py) gains an optional flag
  `include_mutate_checks: bool = False`. When True, decorated functions in
  the module are called (zero-arg) and their returned `SmeltCase` collected,
  alongside the existing module-level instances / `cases` list. Only
  `smelt doctor` passes True; `smelt run` / `smelt evaluate` keep today's
  behavior (never execute functions at import). Undecorated function-style
  cases are never collected — consistent with today.
- `guards` values: `"section:<heading>"` or `"reference:<path>"`.
- Per-case attribution: for each decorated case, doctor runs ONLY its
  declared mutant(s) against that case alone (no short-circuit). If the case
  fails to kill its declared mutant → reported as `false guard`.
- **Dangling guards**: doctor validates every `guards` reference against the
  skill before running (the section/reference must exist). Dangling claims
  are listed under case-suite health and make the run exit 1. (`smelt
  validate` is NOT involved — it never loads case files.)
- Undecorated collected cases still participate in suite-level mutation
  (default safety net).

### R6. Judge canary (`smelt/challenge/canary.py`)

- Built-in bad trace: `fixed_agent` producing an off-topic answer plus a
  hallucinated tool result, judged by the doctor agent.
- Built-in criteria (fixed v1 text): "the answer must address the user's
  actual request and must only use information returned by tool calls;
  inventing tool results or answering a different question is a failure".
- Verdict: `calibrated` if judge score < 0.5, else `miscalibrated`.
- If the loaded cases use no `llm_judge` assertions, canary still runs but
  the report notes it is advisory.

### R7. DoctorReport + output

- Terminal output (default), plus `--output` (.json → JSON, else markdown).
  No implicit file write — reports are only persisted when `--output` is given.
- Report content: case-suite health (cases tested, mutants, per-mutant
  verdicts, surviving mutants with a suggested case idea each, false guards,
  dangling guards), judge health (canary score + verdict), overall
  `ok: bool`, and token usage summed from traces when available.
- Markdown table per mutant: mutant / verdict / killed-by.

### R8. evaluate challenge part — API and config

`evaluate.py`:

- `evaluate_skill(...)` gains `challenger: LLMClient | None = None`. Falls
  back to `judge` when absent (report notes the same-model caveat).
- Builder gains `challenge_enabled: bool = True`, `challenge_rounds: int = 1`,
  `challenge_probes: int = 8`, with `.with_challenge(*, enabled=True,
  rounds=None, probes=None)`.
- CLI `smelt evaluate`: `--no-challenge`, `--challenge-rounds`,
  `--challenge-probes`, plus `--challenger-*` quartet flags.
- Missing challenger+judge → challenge section `skipped` with reason
  (graceful, consistent with writing/suggestions).
- Deterministic backchannel: if cases exist and ALL are deterministic
  (ScriptedLLM / fixed_agent), challenge is `skipped (deterministic
  backend)`. With zero cases this check does not apply — challenge runs
  from probes alone (probes never depend on existing cases).

### R9. Probe generation and execution (`smelt/challenge/probes.py`)

- One challenger call per round returns up to `probes` probe specs as JSON:
  `{type: hard-trigger|no-trigger|distractor, trigger: str, expectation:
  str, fixture: {path: content} | null}`.
- Each probe runs one agent loop (`times=1`) against the skill. The probed
  agent is auto-bound exactly like `_bind` does for cases:
  `smelt_agent(skill, llm=agent_llm or judge, tools=tools)` — the challenger
  never drives the probed agent (attacker/defender separation). If neither
  `agent_llm` nor `judge` is available, challenge is `skipped` with reason.
  - hard-trigger → expect skill behavior (assertion derived from
    `expectation` via judge);
  - no-trigger → expect restraint (no skill-relevant tool calls, judged);
  - distractor → materialize fixture into the workspace, expect the answer
    is not diverted (judged with trace).
- A probe that fails its expectation is a **break**; recorded with trigger,
  expectation, trace summary, and a suggested `new_case(...)` snippet.
- Round ≥2: challenger sees surviving probes and adapts.
- Unparseable/failed probes are recorded as errors, never crash the run.

### R10. Challenge reporting — advisory only

- `SkillEvaluation` gains `challenge: ChallengeResult | None`, serialized in
  `to_dict()`/`to_markdown()` as its own section (probe table:
  type / trigger / verdict / note).
- `overall_score` weights unchanged; challenge contributes nothing unless
  the user opts in: `with_weights()` gains an optional `challenge: float |
  None = None` keyword (existing `behavior/writing/lint` signature
  unchanged). Document the instability caveat on that parameter.
- `compare()` diffs challenge sections informationally (breaks count Δ);
  never counts as regression.

### R11. Exports and docs

- `smelt/__init__.py` exports: `doctor`, `mutate_check`, `LLMConfig`,
  `LLMConfigError`.
- README gains: a "smelt doctor" section (what it checks, exit codes, CI
  snippet), a "challenge" subsection under evaluate, and the role-based
  config table (replacing/extending the current SMELT_* list).
- CHANGELOG entry.

### R12. Testing

- Unit tests for each mutation operator (parse → mutate → assert text).
- Kill-verdict tests with `fixed_agent`/`ScriptedLLM` baselines (deterministic
  scores in/out of the noise band).
- Decorator registry tests; dangling-guard detection tests.
- Canary tests with a `ScriptedLLM` judge returning high/low scores.
- Doctor CLI tests: exit 0/1/2 paths, missing-config error message.
- Challenge tests with `ScriptedLLM` challenger emitting canned probes +
  `fixed_agent`/`ScriptedLLM` agents; degradation paths (no challenger,
  deterministic backend, zero cases).
- Env fallback matrix tests for `LLMConfig.from_role` (explicit > role env >
  legacy env; model required, api_key optional).

## Phasing hint for the plan

The spec is cohesive but large. A plan may split implementation into two
phases: **P1 = R1 + R2–R7** (config + doctor) and **P2 = R8–R10** (evaluate
challenge), with R11–R12 spread across both. Each phase must leave the test
suite green on its own.
