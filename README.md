# Smelt

[![CI](https://github.com/ProjAnvil/Smelt/actions/workflows/ci.yml/badge.svg)](https://github.com/ProjAnvil/Smelt/actions/workflows/ci.yml)

**Smelt** — a behavior-verification framework for agent skills. Put a skill in
the furnace: **given** contexts and an agent, **when** a trigger fires,
**then** assert on the outcome. One consistent yardstick to verify that each
version of a skill behaves better than the last.

Toolchain: uv + Python ≥ 3.11, zero core dependencies.

## Core concept: given / when / then

```python
from smelt import new_case, context, smelt_agent, text, tool_call, text_similar

result = (
    new_case("commit-skill")
    # given: contexts (fixture files, env vars, background prompts)
    .given(context(files="fixtures/dirty_repo", prompt="the user is refactoring"))
    # given: an agent (smelt's built-in LLM tool-loop agent; bring your own too)
    .given(smelt_agent("skills/commit", llm=my_llm, tools=[git_tool]))
    # when: a trigger — a sentence, or a directory of files
    .when(text("commit these changes for me"))
    # then: assertions — commands, JSON schema, strings, similarity to a reference
    .then(tool_call("run_command", args={"cmd": "git status"}))
    .then(text_similar("changes reviewed and committed", threshold=0.8))
    .run()
)
result.assert_passed()  # plugs straight into pytest
```

The fluent API is fully immutable (frozen dataclasses); baseline cases are safe
to derive:

```python
base = new_case("commit").given(smelt_agent(...))
case_a = base.when(text("commit this")).then(tool_call("git"))
case_b = base.when(text("leave it alone")).then(no_tool_call("git"))
```

### Alternative entry: agent-first + fragmented given

`smelt_agent` itself is an entry point; skill / llm / tools can be declared in
segments and are assembled at run time:

```python
from smelt import smelt_agent, skill, llm, tools

result = (
    smelt_agent.new_case("commit")
    .given(skill("skills/commit"))        # or skill(prompt="You are...")
    .given(llm(my_llm))                   # any LLMClient; required
    .given(tools(read_file, write_file))  # repeat to accumulate
    .when(text("commit my changes"))
    .then(tool_call("run_command"))
    .run()
)
```

### Global default thresholds

`text_similar` and `llm_judge` thresholds have defaults (initially 0.8) that can
be tuned project-wide; an explicit threshold always wins:

```python
import smelt

smelt.configure(text_similar_threshold=0.7, llm_judge_threshold=0.75)
```

### .env and SMELT_* variables

Smelt reads credentials and endpoints from a `.env` file — the location is
specified in code (default `.env` in the working directory):

```python
import smelt

smelt.configure(env_file="config/smelt.env")  # or None to disable loading
smelt.load_env()                              # apply now; otherwise auto-loaded on first use
```

```dotenv
# config/smelt.env
SMELT_API_KEY=sk-...
SMELT_BASE_URL=https://api.moonshot.cn/v1
SMELT_JUDGE_MODEL=kimi-k2
```

All variables carry the `SMELT_` prefix. `OpenAIChatClient` falls back to
`SMELT_API_KEY` / `SMELT_BASE_URL` when the arguments are omitted (explicit
arguments win), and `smelt evaluate` falls back to `SMELT_JUDGE_MODEL` when
`--judge-model` is absent. Variables already present in the shell environment
are never overridden by the file.

## given

| helper | description |
|---|---|
| `context(files=..., prompt=..., env=..., vars=...)` | fixture files/dirs copied into an isolated workspace; env injected into tool execution; prompt appended to the system prompt; stackable and merged |
| `reference(path)` / `reference_folder(dir)` | mount skill resources (references/ scripts/ assets/) into the workspace **and register them** — declaration doubles as registration for coverage reporting |
| `smelt_agent(skill, llm=..., tools=[...])` | built-in agent: loads SKILL.md as the system prompt, runs the "LLM → tool → feedback" loop |
| `fixed_agent(output, tool_calls=[...])` | fixed-output agent: replays a preconfigured trace without an LLM — for assertion self-tests and baseline regression |
| custom | implement the `Agent` protocol (`run(ctx, input) -> Trace`) to plug into every assertion |

## when

| helper | description |
|---|---|
| `text("...")` | a one-sentence trigger, delivered as the user message |
| `directory("path/")` | a directory trigger: contents materialized into the workspace, file listing rendered into the user message |

## then (each assertion scores independently with its own threshold)

| helper | scoring |
|---|---|
| `tool_call(name, args={...})` | exact argument-subset match 1.0; name only 0.5; never called 0 |
| `no_tool_call(name)` | negative assertion |
| `output_equals(s)` / `output_contains(*s)` | exact / per-fragment ratio |
| `text_similar(reference, threshold=0.8, scorer=...)` | similarity to the reference; difflib by default, inject an embedding scorer if you like |
| `json_output(schema, contains={...})` | parseable 0.4 + schema 0.4 + field subset 0.2; full JSON Schema with `smelt[json]` installed |
| `llm_judge(judge, criteria=..., reference=..., threshold=0.8)` | LLM-as-judge: a judge model scores the output 0~1 against a rubric |
| `turns_used(max=N)` / `tool_budget(name, max=N)` / `wall_time(max_seconds=S)` | budget gates: agent loop turns / tool call count / wall-clock time — deterministic, binary |
| `reference_read(path)` / `no_reference_read(path)` | reference reach / restraint — tool-name-agnostic path matching (absolute and `../` spellings detected) |
| `reference_untouched(path, source=...)` | ablation validity: proves via content fingerprint that the reference never entered the context |

### Testing references (progressive disclosure level 3)

Skills with depth in `references/` need three distinct probes:

```python
# reach — the task needs it, so the agent must consult it
.then(reference_read("references/endpoints.md"))
# restraint — not needed here, so it stays unread
.then(no_reference_read("references/performance.md"))
# ablation validity — in a WITHOUT arm, prove the content never leaked in
.then(reference_untouched("references/endpoints.md", source="skills/reddit"))
```

**With/without ablation** (measuring what a reference contributes): the without
arm simply does not mount the reference — never mount an empty file instead
(empty ≠ absent; it confounds the measurement). For the cleanest counterfactual,
also drop the pointer sentence from the skill prompt via `skill(prompt=...)`.
Diff the two arms with `compare()` to get the lift. Content-level mocks for
robustness testing use `mock_tool(read_file, {"references/x.md": ""})`.

`evaluate_skill` reports a **Reference Coverage** section: every referenced
file (registered by `reference*` declarations ∪ scanned from SKILL.md) marked
reached/unreached across all behavior cases; `compare()` surfaces
reached→unreached transitions between versions.

### Budget assertions (operating envelopes)

Quality can pass while the run is unaffordable — the "budget burner" failure
mode. Budget assertions are deterministic gates on the trace (no LLM call),
catching versions that get slower or more expensive:

```python
.then(turns_used(max=5))                 # 3 turns vs 8 turns no longer score alike
.then(tool_budget("run_command", max=2)) # or tool_budget(max=4) for the total
.then(wall_time(max_seconds=10))
```

The runner times every run (`trace.wall_time_s`) and agents report their turn
count (`trace.turns`); when the LLM client reports token usage it is summed
into `trace.metadata["token_usage"]`. Because budgets are ordinary
expectations, they flow into repeated sampling, compare(), and CI gates like
any other assertion.

Case score = mean of assertion scores; an assertion passes when `score >= threshold`.
Custom assertions implement the `Expectation` protocol (`evaluate(trace) -> ExpectationResult`).

## Repeated sampling (mean ± std)

LLM agents are stochastic — a single run's score is noise, and noise can be
larger than the difference between two skill versions. Repeat a case and the
score becomes the mean across runs, with the spread (std) reported alongside:

```python
result = (
    new_case("commit")
    .given(smelt_agent("skills/commit", llm=real_llm, tools=[git_tool]))
    .when(text("commit my changes"))
    .then(tool_call("run_command"))
    .repeat(3)          # or: .run(times=3)
    .run()
)
result.score       # mean of 3 run scores (a crashed run counts as 0)
result.score_std   # spread — large std means "result unstable, don't compare"
result.run_scores  # per-run scores, e.g. [1.0, 0.5, 1.0]
result.pass_hat    # pass^k reliability: True only if EVERY run passed
```

The mean flatters an agent that succeeds sometimes; pass^k (tau-bench's
reliability metric) answers "can this skill be trusted every single time".
Both are shown in reports: `0.67 ±0.47 (n=3)  pass^3 ✘` means "usually works,
but not reliably" — a signal the mean alone would hide.

Aggregation: case score = mean of per-run case scores; each expectation's
score = mean across the runs that evaluated it; pass = mean >= threshold.
Configuration errors (missing agent/trigger) never repeat; runtime errors
count as 0 and are listed as run errors. HTML / terminal / markdown reports
all show `±std · n=N` next to repeated scores.

## LLM clients

```python
from smelt import ScriptedLLM, LLMResponse, OpenAIChatClient

# Deterministic replay (CI default): emit canned responses in order
llm = ScriptedLLM([
    LLMResponse.call("run_command", {"cmd": "git status"}),
    LLMResponse.say("status checked"),
])

# Real models (regression / comparing skill versions): OpenAI-compatible API
llm = OpenAIChatClient(model="kimi-k2", base_url="https://api.moonshot.cn/v1")
```

## LLM-as-judge

`llm_judge` lets another model grade the output: a judge model scores 0~1
against a rubric or reference answer, passing at the threshold. The judge is any
`LLMClient` and may differ from the agent under test (e.g. a stronger model):

```python
from smelt import llm_judge, new_case, smelt_agent, text

result = (
    new_case("refusal-quality")
    .given(smelt_agent("skills/refund", llm=agent_llm, tools=[...]))
    .when(text("I refuse to return it, just give me the money"))
    .then(llm_judge(
        judge_llm,
        criteria="must politely decline the non-compliant request and offer a compliant alternative",
        threshold=0.7,
        include_trace=True,  # hand the tool-call process to the judge too
    ))
    .run()
)
```

The judge sees the task input (the when-trigger) by default — grading an
answer without seeing the question inflates scores and blurs version
differences; `include_input=False` opts out. With `include_trace=True` the
judge sees tool calls **including their results**, so it can tell whether the
agent actually used a tool's output or hallucinated from memory. Keep the
judge at temperature 0 (`OpenAIChatClient`'s default), and preferably a
different model family than the agent under test (judges rate their own
family's output higher).

### Per-dimension judging

For a fuller picture of *where* a skill is weak, grade independent dimensions
— each in a **separate** judge call on a categorical 0|1|2 (fail/partial/pass)
scale, averaged into the expectation score:

```python
.then(llm_judge(
    judge_llm,
    dimensions=[
        "tool selection (right tool, or correctly none)",
        "argument correctness (arguments semantically match the request)",
        "result utilization (the answer actually uses what tools returned)",
        "trajectory efficiency (no loops or redundant calls)",
        "error recovery (recovers sensibly from tool errors)",
    ],
    threshold=0.7,
    include_trace=True,  # dimensions like result utilization need the tool results
))
```

One call per dimension prevents anchor bleed (a strong first dimension
dragging the others up) and makes regressions attributable — the result's
`details["dimensions"]` carries each dimension's level and evidence-based
reason. Scoring rationale precedes the score in every judge prompt
(reason-first judging), and all prompts state length neutrality explicitly.

Judge parse failures, out-of-range scores, and call failures all score 0 safely
with an explanatory message — they never crash the case.

## langchain integration (optional)

The core stays dependency-free. With existing langchain model configs, install
`smelt[langchain]` and bridge via `LangChainLLM` (langgraph intentionally unused —
judge and agent both need a single completion, not a graph runtime):

```python
from langchain_openai import ChatOpenAI
from smelt.given.agents.langchain_llm import LangChainLLM

llm = LangChainLLM(ChatOpenAI(model="kimi-k2", base_url="https://api.moonshot.cn/v1"))
# identical to ScriptedLLM / OpenAIChatClient: feed smelt_agent, or use as judge
```

### Provider factory (openai / anthropic, base_url mode)

Or let smelt construct the chat model — provider mode, optional base_url
(provider default when omitted), api_key; three config keys in your `.env`:

```dotenv
SMELT_LLM_PROVIDER=anthropic        # or openai (default)
SMELT_BASE_URL=https://proxy...     # optional; omitted → provider default endpoint
SMELT_API_KEY=sk-...
```

```python
llm = LangChainLLM.from_env("claude-sonnet-4-5")
# or explicitly (explicit args win over the env file):
llm = LangChainLLM.from_provider("openai", "kimi-k2", base_url="https://api.moonshot.cn/v1")
```

Requires `smelt[langchain-openai]` or `smelt[langchain-anthropic]`.

## Comprehensive evaluation & reports (evaluate)

Produce a full evaluation report for a skill: **behavior score** (then-case
results) + **writing score** (per-dimension LLM review of SKILL.md) + **lint
score** (static checks), weighted into an overall grade (A–F), with improvement
suggestions generated code-first from lint findings (the judge LLM only adds
semantic suggestions on top). Fully code-driven:

```python
from smelt import evaluate_skill, new_case, text, tool_call

report = (
    evaluate_skill("skills/commit", judge=judge_llm, agent_llm=agent_llm, tools=[git_tool])
    .with_cases(
        # unbound cases auto-bind the skill under review — replay one suite across versions
        new_case("commit").when(text("commit my changes")).then(tool_call("run_command")),
    )
    .with_lint()                              # static lint part (on by default)
    .with_writing(dimensions=[...])           # LLM writing review, custom dimensions (on by default)
    .with_suggestions(max_items=5)            # LLM improvement suggestions (on by default)
    .with_weights(behavior=0.5, writing=0.3, lint=0.2)  # adjustable weights
    .with_times(3)                            # repeat count for behavior cases (see below)
    .run()
)
print(report.overall_score, report.grade)     # 89.5 'B'
report.save("reports/commit.md")              # extension picks the format; .json → JSON
```

**Behavior cases repeat automatically.** Without an explicit setting, each
case runs 3 times and scores aggregate as mean ± std — that is what makes two
skill versions comparable. Cases driven by deterministic backends
(`fixed_agent`, `ScriptedLLM`) auto-degrade to a single run, so CI replay
stays cheap. Precedence: `.with_times(n)` > per-case `.repeat(n)` > auto.
The markdown/JSON reports carry `runs`, `run_scores` and `score_std` per case.

Report sections: overview (overall/grade/parts) → behavior details (per-then
score vs threshold) → static lint table → writing review table (dimension /
score / comment) → prioritized suggestions. Without a judge, the writing review
is skipped and suggestions are derived from lint findings alone, so a checklist
report is always produced.

### Comparing skill versions (compare)

`compare()` diffs two evaluations — SkillEvaluation objects, payload dicts, or
saved `.json` reports — and tells you whether the numbers *mean* something:

```python
from smelt import compare

diff = compare("reports/skill-v1.json", "reports/skill-v2.json")
print(diff.to_markdown())       # per-case table: baseline / candidate / Δ / verdict
diff.assert_no_regression()     # CI gate: raises on any regressed case
```

A case's score change is **significant only when |Δ| exceeds the noise band**
`max(min_delta, 2σ)` — σ combines both versions' per-run spreads, so a 3-point
"bump" inside the band is reported as `unchanged`, not progress. Independently
of the mean, a **pass^k drop (True → False) is flagged as a reliability
regression** — the version still scores fine on average but no longer works
every time. Verdicts: `improved` / `regressed` / `unchanged` / `added` /
`removed`; a candidate-side runtime error on an existing case is a regression.

CLI equivalent (exit 1 on regression — drop straight into CI):

```bash
smelt evaluate skills/commit --cases cases.py --output reports/v1.json ...
# ...edit the skill...
smelt evaluate skills/commit --cases cases.py --output reports/v2.json ...
smelt compare reports/v1.json reports/v2.json --min-delta 0.05
```

CLI equivalent:

```bash
smelt evaluate skills/commit --cases cases.py \
    --judge-model kimi-k2 --base-url https://api.moonshot.cn/v1 \
    --output reports/commit.md --fail-under 80
```

## Per-case reports (.report / .report_cli)

Every case chain can end in a report:

```python
result = (
    new_case("commit")
    .given(smelt_agent(...))
    .when(text("commit my changes"))
    .then(tool_call("run_command"))
    .report()        # writes a self-contained HTML report to .smelt/reports/
)
result.report_path  # ".smelt/reports/commit-20261006-091500.html" (+ commit-latest.html)
result.assert_passed()  # still pytest-compatible

# or straight to the terminal:
new_case("commit").given(...).when(...).then(...).report_cli()
```

- `.report(output_dir=..., quiet=...)` — dark, self-contained HTML: status badge,
  per-expectation score bars, tool-call trace, final output; a `<slug>-latest.html`
  pointer always tracks the newest run
- `.report_cli(quiet=...)` — formatted terminal report with score bars and the
  tool-call trace
- Invalid cases (missing agent/trigger) produce an error report instead of raising

## Suite reports (multi-case overview)

Run a whole battery and get an `index.html` overview — pass rate, score
distribution, per-case pages linked:

```python
from smelt import suite

result = (
    suite("commit-skill v1.2")
    .add(case_a, case_b, case_c)
    .report()   # .smelt/reports/suites/commit-skill-v1-2-<ts>/index.html + one page per case
)
result.assert_passed()

suite("commit-skill v1.2").add(...).report_cli()  # compact terminal summary
```

Layout: `suites/<slug>-<timestamp>/index.html` with `<case>.html` siblings, plus a
`suites/<slug>-latest/` copy that always tracks the newest run — history is kept
per timestamp, ideal as a CI artifact.

## Running

```bash
uv sync --extra dev

# pytest (recommended; the fixture self-registers)
uv run pytest

# CLI: execute case files directly (module-level new_case(...) or cases = [...])
uv run smelt run examples/cases/demo_cases.py -v

# budget assertions demo: efficient vs wasteful agents (the wasteful case fails on purpose)
uv run smelt run examples/cases/budget_cases.py -v

# static lint: structure / trigger words / asset references for SKILL.md
uv run smelt validate examples --fail-under 80
```

Two assertion styles under pytest:

```python
def test_commit():
    new_case("commit").given(...).when(...).then(...).run().assert_passed()

def test_commit_fixture(smelt):  # the smelt fixture comes from the pytest plugin
    smelt.check(new_case("commit").given(...).when(...).then(...))
```

## Runner guardrails

| shape | behavior |
|---|---|
| no given (no agent/fragments), straight to when/then | ✘ fails with "missing agent" |
| given with context only (no agent) | ✘ fails with "missing agent" |
| fragmented given without llm | ✘ fails with "missing llm fragment" |
| given without when | ✘ fails with "missing trigger" |
| given + when, no then | ✔ runs; empty assertions pass vacuously (flow check) |
| agent crashes at run time | ✘ error converges into `CaseResult.error` |
| double agent / double when / wrong types | build-time error, raised immediately |

## Project layout

```
src/smelt/
├── given/            # the precondition side
│   ├── context.py    #   context(): fixtures/env/prompt → isolated workspace
│   ├── fragments.py  #   skill()/llm()/tools() fragments and assembly
│   └── agents/       #   Agent protocol, SmeltAgent, FixedAgent, LLMClient, LangChainLLM
├── when/             # the trigger side: text / directory
├── then/             # the assertion side: tool_call / text_similar / json_output / llm_judge / ...
├── case.py           # SmeltCase: fluent immutable given/when/then
├── runner.py         # guardrails → materialize context → run agent → evaluate
├── report/           # reports: per-case HTML/text + suite overview (index.html)
├── evaluate.py       # evaluate_skill: behavior + writing + lint review and reports
├── config.py         # global default thresholds (smelt.configure)
├── tools.py          # @tool: signature annotations → JSON Schema
├── pytest_plugin.py  # the smelt fixture
├── cli.py            # smelt run / validate / evaluate
└── lint/             # static quality lint submodule (former skillcheck)
```

## License

MIT
