# Changelog

All notable changes to Smelt are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/).

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

## [0.4.2] - 2026-10-08

### Fixed

- **lint assets: markdown link targets no longer flagged as missing** —
  `PROSE_PATH_RE` re-matched the target of `[text](references/x.md)` as
  `references/x.md)` (trailing paren included), producing phantom
  `referenced file does not exist` errors for every markdown-linked file.
  Link and image targets are now resolved through a CommonMark parser
  (markdown-it-py), and the prose-path scan runs on plain-text segments and
  code blocks only. Reference-style links are covered as a bonus.
- **lint clarity: `{{...}}` no longer flags template syntax in code blocks** —
  the placeholder regex scanned the raw body, so Vue/Svelte interpolation
  shown in fenced examples read as "unfilled placeholder". The `{{...}}`
  pattern now scans prose segments only; TODO/FIXME-style markers still scan
  the full body, code blocks included.
- **`__version__` synced with pyproject.toml** (0.4.0 → 0.4.1), guarded by a
  test comparing the two.

## [0.4.1] - 2026-10-07

### Fixed

- **Reference coverage: anchor normalization** — `_scan_skill_refs` treated
  `references/x.md#锚点` and `references/x.md` as different entries, inflating
  the coverage table with unreachable phantoms. `normalize_ref_path` now strips
  `#fragment` / `?query` suffixes and URL-decodes (Chinese anchors included),
  so the SKILL.md scan and trace matching share one spelling.
- **SmeltAgent messages now follow the OpenAI wire format** — assistant
  messages carry `tool_calls` entries with `id` / `type` / `function`, and tool
  results pair back via `tool_call_id`. Strict providers (e.g. DeepSeek) no
  longer 422 on `missing field 'id'`. `LangChainLLM._to_langchain` accepts both
  the wire format and the legacy flat shape, reusing existing ids.

### Added

- **`OpenAIChatClient(extra_body={...})`** — verbatim request-body passthrough
  for provider-specific switches, e.g. disabling thinking mode:
  `extra_body={"thinking": {"type": "disabled"}}`.

## [0.4.0] - 2026-10-07

Reference-first testing: skills whose depth lives in `references/` are now
fully testable — reach, restraint, ablation validity, and coverage.

### Added

- **`reference(path)` / `reference_folder(dir)`** given fragments — mount skill
  resources (references/, scripts/, assets/) into the workspace at
  skill-relative paths and register them; declaration doubles as registration.
- **`reference_read(path)` / `no_reference_read(path)`** — tool-name-agnostic
  reach/restraint assertions; absolute-path and `../` escape spellings are
  detected, not missed.
- **`reference_untouched(path, source=...)`** — ablation validity: proves via
  content fingerprint that a reference never entered the context (catches
  detour leaks through shell tools and flags parametric-memory contamination).
- **`no_tool_call(name, args={...})`** — args-level negation.
- **`mock_tool(tool, {path: result})`** — content-level mocking for robustness
  tests (empty/broken/stale reference content); not for ablation.
- **Reference Coverage** in `evaluate_skill` reports (markdown + JSON): every
  referenced file (declared ∪ scanned from SKILL.md) marked reached/unreached;
  `compare()` surfaces reached→unreached transitions as `coverage_changes`.
- Example: `examples/cases/reference_cases.py` (reach / restraint / without-arm).

## [0.3.1] - 2026-10-06

### Added

- **LangChain provider factory** — `LangChainLLM.from_provider(provider, model, base_url=, api_key=)`
  supports `openai` and `anthropic` in base_url mode; `base_url` omitted →
  provider default endpoint; `api_key` falls back to `SMELT_API_KEY`
  (auto-loaded .env), then langchain's own env defaults.
  `LangChainLLM.from_env(model)` reads the three-key config file:
  `SMELT_LLM_PROVIDER` / `SMELT_BASE_URL` / `SMELT_API_KEY`.
- New extras: `smelt[langchain-openai]` / `smelt[langchain-anthropic]`;
  missing packages raise an ImportError naming the extra to install.

## [0.3.0] - 2026-10-06

Statistical rigor release: scores are now trustworthy enough to compare skill
versions, and comparison itself is built in.

### Added

- **Repeated sampling** — `case.repeat(n)` / `run(times=n)`; case score is the
  mean of per-run scores (a crashed run counts as 0), every report shows
  `±std · n=N`. `evaluate_skill` defaults to 3 runs for stochastic agents and
  auto-degrades deterministic backends (`fixed_agent` / `ScriptedLLM`) to 1.
- **pass^k reliability** (tau-bench) — `CaseResult.pass_hat` / `run_passed`;
  surfaced in text, HTML, markdown, and JSON reports. A skill that works "most
  of the time" is now visibly different from one that works every time.
- **LLM-as-judge upgrades** — the judge sees the task input by default
  (`include_input=False` opts out); `include_trace=True` now includes tool
  *results* (groundedness); prompts are reason-first with explicit length
  neutrality; new `dimensions=[...]` mode grades each dimension in a separate
  judge call on a categorical 0|1|2 scale (per-dimension verdicts in
  `details["dimensions"]`).
- **`.env` support** — `smelt.configure(env_file=...)` / `smelt.load_env()`;
  `SMELT_API_KEY` / `SMELT_BASE_URL` fallbacks in `OpenAIChatClient`,
  `SMELT_JUDGE_MODEL` in the CLI; existing shell environment always wins.
- **`compare()`** — version-to-version diff of two evaluations (objects,
  dicts, or saved `.json` reports) with a significance band
  `max(min_delta, 2σ)`, pass^k reliability regression detection, and
  `assert_no_regression()`. CLI: `smelt compare v1.json v2.json` exits 1 on
  regression.
- **Budget assertions** (operating envelopes) — `turns_used(max=N)`,
  `tool_budget(name, max=N)`, `wall_time(max_seconds=S)`; deterministic gates
  catching the "budget burner" failure mode. Traces now carry `turns` and
  `wall_time_s`; `LLMResponse.usage` accumulates provider token counts into
  `trace.metadata["token_usage"]`.
- Example: `examples/cases/budget_cases.py` (efficient vs wasteful agents).

### Notes

- Pairwise judging was evaluated and deliberately **not** added: evidence shows
  pairwise protocols flip ~35% under distractor features vs ~9% for absolute
  scoring on objective tasks; the pointwise + significance-band pipeline
  covers version comparison with per-dimension diagnosability.

## [0.2.0] - 2026-10-06

- Suite reports, per-case HTML/terminal reports, GitHub Actions CI
  (ruff + pytest, 90% coverage gate, Python 3.11–3.13).

## [0.1.0] - 2026-10-05

- Initial release: given/when/then behavior verification for agent skills,
  `evaluate_skill` (behavior + writing + lint), LLM-as-judge, static lint
  (former skillcheck), pytest plugin, CLI (`run` / `validate` / `evaluate`).
