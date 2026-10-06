# Smelt

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

## given

| helper | description |
|---|---|
| `context(files=..., prompt=..., env=..., vars=...)` | fixture files/dirs copied into an isolated workspace; env injected into tool execution; prompt appended to the system prompt; stackable and merged |
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

Case score = mean of assertion scores; an assertion passes when `score >= threshold`.
Custom assertions implement the `Expectation` protocol (`evaluate(trace) -> ExpectationResult`).

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

## Comprehensive evaluation & reports (evaluate)

Produce a full evaluation report for a skill: **behavior score** (then-case
results) + **writing score** (per-dimension LLM review of SKILL.md) + **lint
score** (static checks), weighted into an overall grade (A–F), with LLM-generated
improvement suggestions from all the evidence. Fully code-driven:

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
    .run()
)
print(report.overall_score, report.grade)     # 89.5 'B'
report.save("reports/commit.md")              # extension picks the format; .json → JSON
```

Report sections: overview (overall/grade/parts) → behavior details (per-then
score vs threshold) → static lint table → writing review table (dimension /
score / comment) → prioritized suggestions. Without a judge, writing and
suggestions degrade gracefully to "skipped" without blocking other parts.

CLI equivalent:

```bash
smelt evaluate skills/commit --cases cases.py \
    --judge-model kimi-k2 --base-url https://api.moonshot.cn/v1 \
    --output reports/commit.md --fail-under 80
```

## Running

```bash
uv sync --extra dev

# pytest (recommended; the fixture self-registers)
uv run pytest

# CLI: execute case files directly (module-level new_case(...) or cases = [...])
uv run smelt run examples/cases/demo_cases.py -v

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
├── evaluate.py       # evaluate_skill: behavior + writing + lint review and reports
├── config.py         # global default thresholds (smelt.configure)
├── tools.py          # @tool: signature annotations → JSON Schema
├── pytest_plugin.py  # the smelt fixture
├── cli.py            # smelt run / validate / evaluate
└── lint/             # static quality lint submodule (former skillcheck)
```

## License

MIT
