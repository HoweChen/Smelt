# Changelog

All notable changes to Smelt are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/).

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
