# smelt doctor + evaluate challenge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the critique loop to Smelt — role-based LLM config, `smelt doctor` (mutation check + judge canary), and evaluate's adversarial challenge part.

**Architecture:** New `src/smelt/challenge/` package (mutation / canary / probes / doctor) plus a new `src/smelt/llm_config.py`. Doctor questions the yardstick (cases + judge); challenge questions the skill. LLM critique proposes; deterministic checks (noise-band kill verdicts, fixed_agent canary) dispose.

**Spec:** `docs/superpowers/specs/2026-10-08-doctor-challenge.md` (v2, post-audit)

**Tech Stack:** Python ≥3.11, uv, pytest, zero new core dependencies (anthropic provider stays behind the existing langchain extra).

## Global Constraints

- Zero new core dependencies; provider="anthropic" reuses `smelt[langchain-anthropic]`.
- `SmeltCase` stays a frozen dataclass — never mutate it; use `dataclasses.replace`.
- All mutation happens on a temp-dir copy of the skill; the original directory is never touched.
- Challenge results never enter `overall_score` unless the user passes an explicit `challenge=` weight.
- Missing doctor agent → CLI exit 2 (hard error). Missing challenger in evaluate → graceful `skipped`.
- Valid roles: `judge`, `agent`, `doctor`, `challenger`.
- Run tests with `uv run pytest`; lint with `uv run ruff check src tests`.
- One spec deviation (audited and accepted): case loading for doctor lives in
  `smelt/challenge/doctor.py::load_cases_and_guards`; `cli.py::_load_cases`
  stays untouched (the spec's "optional flag" would change a shared function's
  return shape for one caller — a dedicated loader is cleaner).

---

### Task 1: Role-based LLM configuration (`smelt/llm_config.py`)

**Files:**
- Create: `src/smelt/llm_config.py`
- Modify: `src/smelt/env.py` (module docstring only: mention the role-based
  `SMELT_<ROLE>_<KEY>` quartet as the primary scheme, legacy shared keys as
  common fallback)
- Test: `tests/test_llm_config.py`

**Interfaces:**
- Produces: `LLMConfig` (frozen dataclass: `provider: str = "openai"`, `base_url: str | None`, `api_key: str | None`, `model: str | None`), `LLMConfig.from_role(role, *, provider=None, base_url=None, api_key=None, model=None)`, `LLMConfig.build() -> LLMClient`, `LLMConfigError(ValueError)`. Later tasks (doctor CLI, evaluate CLI) resolve every role through `from_role`.

- [ ] **Step 1: Write the failing test**

```python
"""tests/test_llm_config.py"""
import pytest

from smelt.llm_config import LLMConfig, LLMConfigError


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in list(__import__("os").environ):
        if key.startswith("SMELT_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr("smelt.llm_config._auto_load", lambda: None)


def test_role_env_wins_over_legacy(monkeypatch):
    monkeypatch.setenv("SMELT_DOCTOR_MODEL", "claude-sonnet-4-5")
    monkeypatch.setenv("SMELT_DOCTOR_PROVIDER", "anthropic")
    monkeypatch.setenv("SMELT_API_KEY", "shared-key")
    cfg = LLMConfig.from_role("doctor")
    assert cfg.model == "claude-sonnet-4-5"
    assert cfg.provider == "anthropic"
    assert cfg.api_key == "shared-key"  # legacy shared fallback


def test_explicit_args_win(monkeypatch):
    monkeypatch.setenv("SMELT_DOCTOR_MODEL", "env-model")
    cfg = LLMConfig.from_role("doctor", model="explicit-model")
    assert cfg.model == "explicit-model"


def test_legacy_shared_keys_are_common_fallback(monkeypatch):
    monkeypatch.setenv("SMELT_BASE_URL", "https://proxy.example/v1")
    monkeypatch.setenv("SMELT_LLM_PROVIDER", "openai")
    cfg = LLMConfig.from_role("judge", model="kimi-k2")
    assert cfg.base_url == "https://proxy.example/v1"
    assert cfg.provider == "openai"


def test_unknown_role_rejected():
    with pytest.raises(ValueError, match="unknown role"):
        LLMConfig.from_role("wizard")


def test_build_requires_model():
    with pytest.raises(LLMConfigError, match="model"):
        LLMConfig.from_role("doctor").build()


def test_build_openai_client(monkeypatch):
    pytest.importorskip("openai")
    monkeypatch.setenv("SMELT_DOCTOR_API_KEY", "sk-test")
    client = LLMConfig.from_role("doctor", model="kimi-k2").build()
    from smelt.given.agents.llm import OpenAIChatClient
    assert isinstance(client, OpenAIChatClient)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_llm_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'smelt.llm_config'`

- [ ] **Step 3: Write minimal implementation**

```python
"""src/smelt/llm_config.py
Role-based LLM configuration: every role gets its own
provider / base_url / api_key / model quartet.

Env resolution per key (explicit argument always wins)::

    SMELT_<ROLE>_<KEY>  >  SMELT_<KEY> (legacy shared)  >  provider default

Legacy shared keys: SMELT_LLM_PROVIDER, SMELT_BASE_URL, SMELT_API_KEY.
``model`` has no legacy fallback — except JUDGE, whose historical
SMELT_JUDGE_MODEL is exactly the role-style key.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from smelt.env import _auto_load
from smelt.given.agents.llm import LLMClient

ROLES = ("judge", "agent", "doctor", "challenger")

_LEGACY = {"provider": "SMELT_LLM_PROVIDER", "base_url": "SMELT_BASE_URL", "api_key": "SMELT_API_KEY"}


class LLMConfigError(ValueError):
    """A role's LLM configuration is unusable (model unresolvable)."""


@dataclass(frozen=True)
class LLMConfig:
    provider: str = "openai"  # "openai" | "anthropic"
    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None

    @classmethod
    def from_role(
        cls,
        role: str,
        *,
        provider: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ) -> LLMConfig:
        _auto_load()
        r = role.lower()
        if r not in ROLES:
            raise ValueError(f"unknown role {role!r}: expected one of {ROLES}")
        prefix = f"SMELT_{r.upper()}_"

        def pick(key: str, explicit: str | None, default: str | None = None) -> str | None:
            if explicit is not None:
                return explicit
            return os.environ.get(prefix + key) or os.environ.get(_LEGACY.get(key.lower(), "")) or default

        return cls(
            provider=pick("PROVIDER", provider, "openai"),
            base_url=pick("BASE_URL", base_url),
            api_key=pick("API_KEY", api_key),
            model=model or os.environ.get(prefix + "MODEL"),
        )

    def build(self) -> LLMClient:
        if not self.model:
            raise LLMConfigError(
                f"model not configured: set SMELT_<ROLE>_MODEL in .env or pass an explicit model"
            )
        if self.provider == "openai":
            from smelt.given.agents.llm import OpenAIChatClient

            return OpenAIChatClient(self.model, base_url=self.base_url, api_key=self.api_key)
        if self.provider == "anthropic":
            from smelt.given.agents.langchain_llm import LangChainLLM

            return LangChainLLM.from_provider(
                "anthropic", self.model, base_url=self.base_url, api_key=self.api_key
            )
        raise LLMConfigError(f"unknown provider {self.provider!r}: expected 'openai' or 'anthropic'")
```

Note: `pick` uses `_LEGACY.get(key.lower(), "")`; for key "MODEL" the legacy
lookup yields `os.environ.get("")` → None — harmless, model has no legacy key.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_llm_config.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/smelt/llm_config.py tests/test_llm_config.py
git commit -m "feat: role-based LLM configuration (LLMConfig.from_role)"
```

---

### Task 2: Mutation operators (`smelt/challenge/mutation.py`)

**Files:**
- Create: `src/smelt/challenge/__init__.py` (empty docstring only)
- Create: `src/smelt/challenge/mutation.py`
- Test: `tests/test_mutation.py`

**Interfaces:**
- Produces: `Mutant` (frozen dataclass: `id: str`, `kind: str`, `target: str = ""`), `generate_mutants(skill: Path) -> list[Mutant]`, `apply_mutant(skill: Path, mutant: Mutant, dest_root: Path) -> Path | None`. Task 3+ consume these.

- [ ] **Step 1: Write the failing test**

```python
"""tests/test_mutation.py"""
from pathlib import Path

from smelt.challenge.mutation import Mutant, apply_mutant, generate_mutants

SKILL_MD = """---
name: commit
description: Create git commits when the user asks to save changes
---

# Commit Skill

## Usage
Always run git status before committing.

## Boundaries
Do not commit merge requests. Never force-push.

## Notes
Some neutral line.
"""


def make_skill(tmp_path: Path) -> Path:
    d = tmp_path / "skills" / "commit"
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(SKILL_MD, encoding="utf-8")
    (d / "references").mkdir()
    (d / "references" / "merge.md").write_text("# merge\n", encoding="utf-8")
    return d


def test_generate_mutants_covers_sections_and_references(tmp_path):
    skill = make_skill(tmp_path)
    ids = [m.id for m in generate_mutants(skill)]
    assert "drop_section:Usage" in ids
    assert "drop_section:Boundaries" in ids
    assert "drop_reference:references/merge.md" in ids
    assert "drop_constraint" in ids
    assert "drop_requirement" in ids


def test_drop_section_removes_body(tmp_path):
    skill = make_skill(tmp_path)
    mutant = Mutant(id="drop_section:Boundaries", kind="drop_section", target="Boundaries")
    dest = apply_mutant(skill, mutant, tmp_path / "mutants")
    assert dest is not None
    text = (dest / "SKILL.md").read_text(encoding="utf-8")
    assert "merge requests" not in text
    assert "## Usage" in text  # other sections intact
    assert (skill / "SKILL.md").read_text(encoding="utf-8") == SKILL_MD  # original untouched


def test_drop_reference_deletes_file(tmp_path):
    skill = make_skill(tmp_path)
    mutant = Mutant(id="drop_reference:references/merge.md", kind="drop_reference", target="references/merge.md")
    dest = apply_mutant(skill, mutant, tmp_path / "mutants")
    assert dest is not None
    assert not (dest / "references" / "merge.md").exists()


def test_drop_constraint_removes_prohibition_lines(tmp_path):
    skill = make_skill(tmp_path)
    mutant = Mutant(id="drop_constraint", kind="drop_constraint")
    dest = apply_mutant(skill, mutant, tmp_path / "mutants")
    text = (dest / "SKILL.md").read_text(encoding="utf-8")
    assert "Do not commit" not in text and "Never force-push" not in text
    assert "Always run git status" in text


def test_invalid_mutant_returns_none(tmp_path):
    skill = make_skill(tmp_path)
    mutant = Mutant(id="drop_section:Nope", kind="drop_section", target="Nope")
    assert apply_mutant(skill, mutant, tmp_path / "mutants") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_mutation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'smelt.challenge'`

- [ ] **Step 3: Write minimal implementation**

`src/smelt/challenge/__init__.py`:
```python
"""challenge package: adversarial critique for skills (doctor / probes)."""
```

`src/smelt/challenge/mutation.py`:
```python
"""Mutation check: deliberately break the skill, watch whether cases notice.

Mutants are deterministic text transforms applied to an isolated copy of the
skill — the original directory is never touched. A mutant that cannot be
applied (missing section, no matching sentence, unparsable result) is
reported as invalid and excluded from the mutation score.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from smelt.evaluate import _scan_skill_refs

_CONSTRAINT_RE = re.compile(r"(?i)\b(do not|don't|never|must not)\b|不要|禁止|切勿|不得")
_REQUIREMENT_RE = re.compile(r"(?i)\b(must|always|required|be sure to)\b|必须|务必|总是")
_SECTION_RE = re.compile(r"^##\s+(.+?)\s*$", re.M)


@dataclass(frozen=True)
class Mutant:
    id: str  # "drop_section:Boundaries" / "drop_reference:references/x.md" / "drop_constraint"
    kind: str  # drop_section | drop_reference | drop_constraint | drop_requirement
    target: str = ""


def _skill_dir(skill: Path) -> Path:
    return skill.parent if skill.is_file() else skill


def _sections(text: str) -> list[tuple[str, int, int]]:
    """(title, start, end) per '## ' section; end = next '##' start or EOF."""
    matches = list(_SECTION_RE.finditer(text))
    return [
        (m.group(1), m.start(), matches[i + 1].start() if i + 1 < len(matches) else len(text))
        for i, m in enumerate(matches)
    ]


def generate_mutants(skill: Path) -> list[Mutant]:
    """One mutant per '##' section and per referenced file, plus one
    drop_constraint and one drop_requirement."""
    skill_dir = _skill_dir(skill)
    doc = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
    mutants = [
        Mutant(id=f"drop_section:{title}", kind="drop_section", target=title)
        for title, _, _ in _sections(doc)
    ]
    refs = set(_scan_skill_refs(doc))
    ref_root = skill_dir / "references"
    if ref_root.is_dir():
        refs |= {f"references/{p.name}" for p in sorted(ref_root.iterdir()) if p.is_file()}
    mutants += [Mutant(id=f"drop_reference:{r}", kind="drop_reference", target=r) for r in sorted(refs)]
    mutants.append(Mutant(id="drop_constraint", kind="drop_constraint"))
    mutants.append(Mutant(id="drop_requirement", kind="drop_requirement"))
    return mutants


def _valid(skill_dir: Path) -> bool:
    """The mutated copy still parses: SKILL.md exists with non-empty body."""
    md = skill_dir / "SKILL.md"
    if not md.is_file():
        return False
    body = md.read_text(encoding="utf-8")
    if body.startswith("---"):
        parts = body.split("---", 2)
        body = parts[2] if len(parts) == 3 else ""
    return bool(body.strip())


def apply_mutant(skill: Path, mutant: Mutant, dest_root: Path) -> Path | None:
    """Copy the skill into dest_root/<slug>, apply the mutation, return the copy.
    None when the mutant cannot be applied or leaves an invalid skill."""
    src = _skill_dir(skill)
    slug = re.sub(r"[^\w.-]+", "_", mutant.id)
    dest = dest_root / slug
    shutil.copytree(src, dest, dirs_exist_ok=True)
    md = dest / "SKILL.md"
    text = md.read_text(encoding="utf-8")

    if mutant.kind == "drop_section":
        hit = [s for s in _sections(text) if s[0] == mutant.target]
        if not hit:
            shutil.rmtree(dest)
            return None
        _, start, end = hit[0]
        md.write_text(text[:start] + text[end:], encoding="utf-8")
    elif mutant.kind == "drop_reference":
        target = dest / mutant.target
        if not target.is_file():
            shutil.rmtree(dest)
            return None
        target.unlink()
    elif mutant.kind in ("drop_constraint", "drop_requirement"):
        pattern = _CONSTRAINT_RE if mutant.kind == "drop_constraint" else _REQUIREMENT_RE
        kept = [ln for ln in text.splitlines() if not pattern.search(ln)]
        if len(kept) == len(text.splitlines()):
            shutil.rmtree(dest)
            return None
        md.write_text("\n".join(kept) + "\n", encoding="utf-8")
    else:
        shutil.rmtree(dest)
        return None

    if not _valid(dest):
        shutil.rmtree(dest)
        return None
    return dest
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_mutation.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/smelt/challenge/__init__.py src/smelt/challenge/mutation.py tests/test_mutation.py
git commit -m "feat: deterministic skill mutation operators"
```

---

### Task 3: Kill verdict + mutant runner (`smelt/challenge/mutation.py` continued)

**Files:**
- Modify: `src/smelt/challenge/mutation.py` (append)
- Test: `tests/test_mutation_kill.py`

**Interfaces:**
- Consumes: `Mutant`, `apply_mutant` (Task 2); `CaseResult.score`, `CaseResult.score_std` (smelt.results); `SmeltCase` (smelt.case); `_is_deterministic` (smelt.evaluate).
- Produces: `killed(baseline: CaseResult, mutant: CaseResult, *, min_delta: float = 0.05) -> bool`; `rebind_for_mutation(case: SmeltCase, skill_dir: Path, doctor_llm, tools) -> SmeltCase | None` (None = case cannot perceive mutations). Doctor (Task 6) consumes both.

- [ ] **Step 1: Write the failing test**

```python
"""tests/test_mutation_kill.py"""
from pathlib import Path

from smelt import LLMResponse, ScriptedLLM, fixed_agent, new_case, smelt_agent, text, tool_call
from smelt.challenge.mutation import killed, rebind_for_mutation
from smelt.results import CaseResult, ExpectationResult


def make_result(score: float, std: float = 0.0) -> CaseResult:
    runs = [score - std, score + std] if std > 0 else []
    return CaseResult(case_name="c", expectations=[ExpectationResult("e", score, 0.5)],
                      run_scores=runs)


def test_killed_beyond_noise_band():
    assert killed(make_result(0.9), make_result(0.5))
    assert not killed(make_result(0.9), make_result(0.87))  # within 0.05 floor
    assert not killed(make_result(0.9, std=0.1), make_result(0.75))  # drop 0.15 < 2σ=0.2


def test_killed_never_when_scores_equal():
    assert not killed(make_result(0.8), make_result(0.8))


def test_rebind_fixed_agent_returns_none(tmp_path):
    case = new_case("x").given(fixed_agent("done")).when(text("hi"))
    assert rebind_for_mutation(case, tmp_path, None, ()) is None


def test_rebind_scripted_llm_returns_none(tmp_path):
    case = (new_case("x")
            .given(smelt_agent(tmp_path, llm=ScriptedLLM([LLMResponse.say("ok")])))
            .when(text("hi")))
    assert rebind_for_mutation(case, tmp_path, None, ()) is None


def test_rebind_unbound_case_gets_doctor_agent(tmp_path):
    doctor = ScriptedLLM([LLMResponse.say("ok")])  # stand-in; real use passes a live client
    case = new_case("x").when(text("hi")).then(tool_call("run_command"))
    rebound = rebind_for_mutation(case, tmp_path, doctor, ())
    assert rebound is not None and rebound.agent is not None
    assert str(rebound.agent.skill) == str(tmp_path)


def test_rebind_smelt_agent_keeps_its_llm(tmp_path):
    class FakeLLM:
        def complete(self, messages, tools): ...

    case = (new_case("x").given(smelt_agent("orig/skill", llm=FakeLLM(), tools=()))
            .when(text("hi")))
    rebound = rebind_for_mutation(case, tmp_path, None, ())
    assert rebound is not None
    assert isinstance(rebound.agent.llm, FakeLLM)
    assert str(rebound.agent.skill) == str(tmp_path)
```

Note: `test_rebind_unbound_case_gets_doctor_agent` passes a ScriptedLLM as the
doctor stand-in; `_is_deterministic` must only inspect the case's own agent,
not the doctor fallback — if the implementation makes this test fail by
returning None, that is a bug in the implementation, not the test.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_mutation_kill.py -v`
Expected: FAIL with `ImportError: cannot import name 'killed'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/smelt/challenge/mutation.py`:

```python
# ---------------------------------------------------------------------------
# Kill verdict and case rebinding
# ---------------------------------------------------------------------------

from smelt.case import SmeltCase  # noqa: E402
from smelt.given.agents.smelt import SmeltAgent  # noqa: E402
from smelt.results import CaseResult  # noqa: E402


def killed(baseline: CaseResult, mutant: CaseResult, *, min_delta: float = 0.05) -> bool:
    """A mutant is killed by a case when the score drop exceeds the noise band
    max(min_delta, 2σ) — σ from the baseline's per-run spread (compare() formula)."""
    band = max(min_delta, 2.0 * baseline.score_std)
    return baseline.score - mutant.score > band


def rebind_for_mutation(
    case: SmeltCase,
    skill_dir: Path,
    doctor_llm,
    tools,
) -> SmeltCase | None:
    """Return the case re-pointed at a mutated skill copy; None when the case
    cannot perceive mutations (deterministic backend, custom agent, or
    fragment-bound skill)."""
    from dataclasses import replace

    from smelt.evaluate import _is_deterministic
    from smelt.given.fragments import SkillSpec

    # NOTE: fallback is None on purpose — determinism is a property of the
    # case's own agent/fragments, never of the doctor stand-in LLM.
    if _is_deterministic(case, None):
        return None
    if case.fragments:
        if any(isinstance(f, SkillSpec) for f in case.fragments):
            return None
        return None  # v1: fragment-assembled agents are not rebound
    agent = case.agent
    if agent is None:
        return case.given(SmeltAgent(skill=str(skill_dir), llm=doctor_llm, tools=tools))
    if isinstance(agent, SmeltAgent):
        return replace(
            case,
            agent=SmeltAgent(
                skill=str(skill_dir),
                llm=agent.llm or doctor_llm,
                tools=agent.tools or tools,
                max_turns=agent.max_turns,
                system_prompt=agent.system_prompt,
            ),
        )
    return None  # custom Agent: opaque, cannot rebind
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_mutation_kill.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/smelt/challenge/mutation.py tests/test_mutation_kill.py
git commit -m "feat: kill verdict (noise band) and mutation rebinding"
```

---

### Task 4: `@mutate_check` decorator + guarded case loading

**Files:**
- Create: `src/smelt/challenge/guards.py`
- Test: `tests/test_mutate_check.py`

**Interfaces:**
- Produces: `MutateSpec` (frozen: `guards: tuple[str, ...]`), `mutate_check` decorator (bare or `guards=`), `load_cases_and_guards(path: Path) -> tuple[list[SmeltCase], dict[str, MutateSpec]]`, `find_dangling_guards(guards_map, skill: Path) -> list[str]`. Doctor (Task 6) consumes all three.

- [ ] **Step 1: Write the failing test**

```python
"""tests/test_mutate_check.py"""
from pathlib import Path

from smelt import new_case, text, tool_call
from smelt.challenge.guards import find_dangling_guards, load_cases_and_guards, mutate_check

CASE_FILE = '''
from smelt import new_case, text, tool_call
from smelt.challenge.guards import mutate_check

plain = new_case("plain").when(text("hi")).then(tool_call("run_command"))

@mutate_check(guards="section:Boundaries")
def merge_refusal():
    return new_case("merge-refusal").when(text("merge commit")).then(tool_call("run_command"))

@mutate_check(guards=["reference:references/merge.md"])
def ref_guard():
    return new_case("ref-guard").when(text("x")).then(tool_call("run_command"))
'''


def make_case_file(tmp_path: Path) -> Path:
    f = tmp_path / "cases.py"
    f.write_text(CASE_FILE, encoding="utf-8")
    return f


def make_skill(tmp_path: Path) -> Path:
    d = tmp_path / "skill"
    d.mkdir()
    (d / "SKILL.md").write_text(
        "---\nname: s\n---\n\n# S\n\n## Boundaries\nno merges\n", encoding="utf-8"
    )
    return d


def test_loader_collects_plain_and_decorated(tmp_path):
    cases, guards = load_cases_and_guards(make_case_file(tmp_path))
    names = {c.name for c in cases}
    assert names == {"plain", "merge-refusal", "ref-guard"}
    assert guards["merge-refusal"].guards == ("section:Boundaries",)
    assert guards["ref-guard"].guards == ("reference:references/merge.md",)
    assert "plain" not in guards


def test_dangling_guard_detected(tmp_path):
    from smelt.challenge.guards import MutateSpec

    skill = make_skill(tmp_path)
    guards = {"c": MutateSpec(guards=("section:Nope", "reference:references/gone.md"))}
    dangling = find_dangling_guards(guards, skill)
    assert any("section:Nope" in d for d in dangling)
    assert any("reference:references/gone.md" in d for d in dangling)


def test_valid_guards_not_dangling(tmp_path):
    from smelt.challenge.guards import MutateSpec

    skill = make_skill(tmp_path)
    guards = {"c": MutateSpec(guards=("section:Boundaries",))}
    assert find_dangling_guards(guards, skill) == []
```

(The final test file contains exactly three tests: test_loader_collects_plain_and_decorated,
test_dangling_guard_detected, test_valid_guards_not_dangling.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_mutate_check.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'smelt.challenge.guards'`

- [ ] **Step 3: Write minimal implementation**

```python
"""src/smelt/challenge/guards.py
@mutate_check: declare which skill part a case guards; doctor verifies the claim.

The decorator only MARKS the function (attribute, not a global registry), so
loading stays module-local and order-free. Doctor's loader calls decorated
zero-arg functions to obtain their cases; `smelt run` / `smelt evaluate`
never execute functions at import (unchanged behavior).
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from smelt.case import SmeltCase
from smelt.challenge.mutation import _sections


@dataclass(frozen=True)
class MutateSpec:
    guards: tuple[str, ...] = ()  # "section:<heading>" | "reference:<path>"


def mutate_check(func: Callable | None = None, *, guards: str | Sequence[str] | None = None):
    """Mark a zero-arg function returning a SmeltCase as a mutation-checked case.

    Bare:        @mutate_check
    With claims: @mutate_check(guards="section:Boundaries")
    """
    spec = MutateSpec(
        guards=() if guards is None else (guards,) if isinstance(guards, str) else tuple(guards)
    )

    def deco(f: Callable) -> Callable:
        f._smelt_mutate = spec  # type: ignore[attr-defined]
        return f

    return deco(func) if callable(func) else deco


def load_cases_and_guards(path: Path) -> tuple[list[SmeltCase], dict[str, MutateSpec]]:
    """Import a case file; collect module-level SmeltCase instances / a cases
    list (same rule as cli._load_cases) plus decorated functions (called to
    obtain their case). Returns (cases, guards-by-case-name)."""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load case file: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cases: list[SmeltCase] = []
    guards: dict[str, MutateSpec] = {}
    explicit = getattr(module, "cases", None)
    if explicit is not None:
        cases.extend(c for c in explicit if isinstance(c, SmeltCase))
    else:
        cases.extend(v for v in vars(module).values() if isinstance(v, SmeltCase))
    for v in vars(module).values():
        mark = getattr(v, "_smelt_mutate", None)
        if mark is None or not callable(v):
            continue
        case = v()
        if not isinstance(case, SmeltCase):
            raise TypeError(f"@mutate_check function {v.__name__} must return a SmeltCase")
        cases.append(case)
        if mark.guards:
            guards[case.name] = mark
    return cases, guards


def find_dangling_guards(guards: dict[str, MutateSpec], skill: Path) -> list[str]:
    """Every guard claim must resolve to an existing '##' section or reference
    file. Returns human-readable dangling entries (empty when all resolve)."""
    skill_dir = skill.parent if skill.is_file() else skill
    doc = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
    titles = {t for t, _, _ in _sections(doc)}
    dangling: list[str] = []
    for case_name, spec in guards.items():
        for g in spec.guards:
            kind, _, target = g.partition(":")
            if kind == "section" and target not in titles:
                dangling.append(f"{case_name}: guards missing section {target!r}")
            elif kind == "reference" and not (skill_dir / target).is_file():
                dangling.append(f"{case_name}: guards missing reference {target!r}")
            elif kind not in ("section", "reference"):
                dangling.append(f"{case_name}: unknown guard kind {kind!r} (expected section:/reference:)")
    return dangling
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_mutate_check.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/smelt/challenge/guards.py tests/test_mutate_check.py
git commit -m "feat: @mutate_check guard declarations and guarded case loading"
```

---

### Task 5: Judge canary (`smelt/challenge/canary.py`)

**Files:**
- Create: `src/smelt/challenge/canary.py`
- Test: `tests/test_canary.py`

**Interfaces:**
- Produces: `CanaryResult` (frozen: `score: float`, `reason: str`, `threshold: float = 0.5`; property `calibrated -> bool`), `run_canary(doctor_llm: LLMClient) -> CanaryResult`, `CANARY_THRESHOLD = 0.5`. Consumes `llm_judge` (smelt.then) and `Trace` (smelt.trace).

- [ ] **Step 1: Write the failing test**

```python
"""tests/test_canary.py"""
from smelt import LLMResponse, ScriptedLLM
from smelt.challenge.canary import run_canary


def test_miscalibrated_judge_scores_canary_high():
    judge = ScriptedLLM([LLMResponse.say('{"reason": "fluent answer", "score": 0.9}')])
    result = run_canary(judge)
    assert not result.calibrated
    assert result.score == 0.9


def test_calibrated_judge_scores_canary_low():
    judge = ScriptedLLM([LLMResponse.say('{"reason": "off-topic and invented tool result", "score": 0.1}')])
    result = run_canary(judge)
    assert result.calibrated
    assert result.score == 0.1


def test_judge_failure_counts_as_miscalibrated():
    judge = ScriptedLLM([LLMResponse.say("not json at all")])
    result = run_canary(judge)
    assert not result.calibrated  # judge parse failures score 0... must still be flagged
```

Note on the third test: judge parse failures score 0 with an explanatory
message (then/judge.py contract). A score-0 parse failure looks "calibrated"
but proves nothing. Decide: `run_canary` treats a judge-side error/parse
failure as `calibrated=False` with reason surfaced — implement by checking
`result.message` for the judge failure marker; simplest robust rule:
calibrated only when `0 < score < threshold` OR score == 0 with an empty
message... too fragile. Final rule (implement exactly this):

```python
calibrated = result.score < CANARY_THRESHOLD and "failed" not in result.message.lower() and "parse" not in result.message.lower()
```

Rewrite the third test to match: score 0.0 from parse failure → message
contains "parse" → calibrated False. A genuine 0.0 score from a working judge
contains a normal reason → calibrated True.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_canary.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'smelt.challenge.canary'`

- [ ] **Step 3: Write minimal implementation**

```python
"""src/smelt/challenge/canary.py
Judge canary: feed the judge one deliberately bad trace; a healthy judge
scores it low. Named for the coal-mine canary — when it dies, the air
(the judge's calibration) is toxic and every score it produced is suspect.

Deterministic by construction: the trace is fixed; only the judge under
test is live.
"""

from __future__ import annotations

from dataclasses import dataclass

from smelt.given.agents.llm import LLMClient
from smelt.then.judge import llm_judge
from smelt.trace import Trace

CANARY_THRESHOLD = 0.5

CANARY_INPUT = "Refactor the payment module and summarize what you changed."
CANARY_OUTPUT = (
    "The weather in Paris is nice today. Also, the refund API returned 200, "
    "so I issued the coupon and closed the ticket."
)
CANARY_CRITERIA = (
    "the answer must address the user's actual request and must only use "
    "information returned by tool calls; inventing tool results or answering "
    "a different question is a failure"
)


@dataclass(frozen=True)
class CanaryResult:
    score: float
    reason: str
    threshold: float = CANARY_THRESHOLD

    @property
    def calibrated(self) -> bool:
        flagged = "failed" in self.reason.lower() or "parse" in self.reason.lower()
        return self.score < self.threshold and not flagged


def run_canary(doctor_llm: LLMClient) -> CanaryResult:
    """Judge one built-in bad trace: off-topic answer + hallucinated tool result
    (the trace deliberately contains NO tool calls)."""
    trace = Trace(
        messages=[
            {"role": "user", "content": CANARY_INPUT},
            {"role": "assistant", "content": CANARY_OUTPUT},
        ],
        output=CANARY_OUTPUT,
        tool_calls=[],
        turns=1,
    )
    result = llm_judge(doctor_llm, criteria=CANARY_CRITERIA, include_trace=True).evaluate(trace)
    return CanaryResult(score=result.score, reason=result.message)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_canary.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/smelt/challenge/canary.py tests/test_canary.py
git commit -m "feat: judge canary (built-in bad trace, calibrated verdict)"
```

---

### Task 6: DoctorReport + `doctor()` (`smelt/challenge/doctor.py`)

**Files:**
- Create: `src/smelt/challenge/doctor.py`
- Test: `tests/test_doctor.py`

**Interfaces:**
- Consumes: Tasks 1–5 (`LLMConfig`, mutants, `killed`, `rebind_for_mutation`, `load_cases_and_guards`, `find_dangling_guards`, `run_canary`).
- Produces: `MutantResult` (dataclass: `mutant: Mutant`, `verdict: str`, `killed_by: tuple[str, ...] = ()`, `note: str = ""`), `DoctorReport` (dataclass with `ok` property, `to_dict`, `to_markdown`, `save`), `doctor(case_files, *, skill, doctor=None, min_score=0.8, tools=()) -> DoctorReport`. CLI (Task 7) consumes these.

- [ ] **Step 1: Write the failing test**

Strategy: doctor's case-running seam is injectable so tests never touch an
LLM. `doctor(...)` gains a keyword-only `_run` hook (defaults to the real
runner): `_run(cases: list[SmeltCase], skill_dir: Path) -> list[CaseResult]`
used for BOTH baseline and mutant runs. Tests drive it with canned scores
keyed by skill dir name (baseline dir = the pristine copy; mutant dirs carry
the mutant slug).

```python
"""tests/test_doctor.py"""
from pathlib import Path

from smelt import LLMResponse, ScriptedLLM, fixed_agent, new_case, text, tool_call
from smelt.challenge.doctor import doctor
from smelt.results import CaseResult, ExpectationResult

SKILL_MD = """---
name: commit
---

# Commit

## Boundaries
Do not commit merges.
"""


def make_skill(tmp_path: Path) -> Path:
    d = tmp_path / "skill"
    d.mkdir()
    (d / "SKILL.md").write_text(SKILL_MD, encoding="utf-8")
    return d


def make_case_file(tmp_path: Path) -> Path:
    f = tmp_path / "cases.py"
    f.write_text(
        "from smelt import new_case, text, tool_call\n"
        "c = new_case('c1').when(text('commit')).then(tool_call('run_command'))\n",
        encoding="utf-8",
    )
    return f


def score_run(score: float):
    def _run(cases, skill_dir, times=None):
        return [CaseResult(case_name=c.name,
                           expectations=[ExpectationResult("e", score, 0.5)]) for c in cases]
    return _run


def flat_judge():
    return ScriptedLLM([LLMResponse.say('{"reason": "ok", "score": 0.1}')])


def test_all_mutants_killed(tmp_path):
    # Baseline 1.0; every mutant dir scores 0.0 → all killed → score 1.0, ok.
    def _run(cases, skill_dir, times=None):
        score = 1.0 if skill_dir.name == "pristine" else 0.0
        return score_run(score)(cases, skill_dir, times)

    report = doctor([make_case_file(tmp_path)], skill=make_skill(tmp_path),
                    doctor=flat_judge(), _run=_run)
    assert report.mutation_score == 1.0
    assert report.ok


def test_surviving_mutant_reported(tmp_path):
    # Mutants never change the score → all survive → mutation score 0, not ok.
    report = doctor([make_case_file(tmp_path)], skill=make_skill(tmp_path),
                    doctor=flat_judge(), _run=score_run(1.0))
    assert report.mutation_score == 0.0
    assert not report.ok
    assert all(m.verdict == "survived" for m in report.mutants)


def test_deterministic_cases_mark_mutation_na(tmp_path):
    f = tmp_path / "cases.py"
    f.write_text(
        "from smelt import new_case, fixed_agent, text\n"
        "c = new_case('fx').given(fixed_agent('done')).when(text('hi'))\n",
        encoding="utf-8",
    )
    report = doctor([f], skill=make_skill(tmp_path), doctor=flat_judge(), _run=score_run(1.0))
    assert report.mutation_score is None
    assert any("deterministic" in n for n in report.notes)


def test_canary_miscalibrated_marks_not_ok(tmp_path):
    bad_judge = ScriptedLLM([LLMResponse.say('{"reason": "great", "score": 0.95}')])
    report = doctor([make_case_file(tmp_path)], skill=make_skill(tmp_path),
                    doctor=bad_judge, _run=score_run(1.0))
    assert report.canary is not None and not report.canary.calibrated
    assert not report.ok
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_doctor.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'smelt.challenge.doctor'`

- [ ] **Step 3: Write minimal implementation**

```python
"""src/smelt/challenge/doctor.py
smelt doctor: health check for the yardstick — case suite (mutation check)
and judge (canary). Exit-code mapping lives in the CLI; here ok is a property.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from smelt.case import SmeltCase
from smelt.challenge.canary import CanaryResult, run_canary
from smelt.challenge.guards import find_dangling_guards, load_cases_and_guards
from smelt.challenge.mutation import Mutant, apply_mutant, generate_mutants, killed, rebind_for_mutation
from smelt.given.agents.llm import LLMClient
from smelt.results import CaseResult


@dataclass(frozen=True)
class MutantResult:
    mutant: Mutant
    verdict: str  # killed | survived | invalid
    killed_by: tuple[str, ...] = ()
    note: str = ""


@dataclass
class DoctorReport:
    skill_path: str
    cases_tested: int
    mutants: list[MutantResult] = field(default_factory=list)
    mutation_score: float | None = None  # None = N/A (deterministic backend)
    false_guards: list[str] = field(default_factory=list)
    dangling_guards: list[str] = field(default_factory=list)
    canary: CanaryResult | None = None
    notes: list[str] = field(default_factory=list)
    min_score: float = 0.8
    token_usage: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        if self.false_guards or self.dangling_guards:
            return False
        if self.canary is not None and not self.canary.calibrated:
            return False
        if self.mutation_score is not None and self.mutation_score < self.min_score:
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill": self.skill_path,
            "ok": self.ok,
            "cases_tested": self.cases_tested,
            "mutation_score": self.mutation_score,
            "min_score": self.min_score,
            "mutants": [
                {"id": m.mutant.id, "verdict": m.verdict, "killed_by": list(m.killed_by), "note": m.note}
                for m in self.mutants
            ],
            "false_guards": self.false_guards,
            "dangling_guards": self.dangling_guards,
            "canary": (
                {"score": self.canary.score, "calibrated": self.canary.calibrated, "reason": self.canary.reason}
                if self.canary else None
            ),
            "notes": self.notes,
            "token_usage": self.token_usage,
        }

    def to_markdown(self) -> str:
        lines = [f"# smelt doctor: {self.skill_path}", ""]
        lines.append(f"Cases tested: {self.cases_tested}")
        if self.mutation_score is None:
            lines.append("Mutation check: N/A")
        else:
            lines.append(f"Mutation score: {self.mutation_score:.0%} (gate {self.min_score:.0%})")
        lines += ["", "| Mutant | Verdict | Killed by |", "|---|---|---|"]
        for m in self.mutants:
            lines.append(f"| {m.mutant.id} | {m.verdict} | {', '.join(m.killed_by) or '-'} |")
        for m in self.mutants:
            if m.verdict == "survived":
                lines.append(f"\n⚠ surviving mutant `{m.mutant.id}` — {m.note}")
        for g in self.false_guards:
            lines.append(f"\n⚠ false guard: {g}")
        for g in self.dangling_guards:
            lines.append(f"\n⚠ dangling guard: {g}")
        if self.canary is not None:
            state = "calibrated ✔" if self.canary.calibrated else "MISCALIBRATED ✘"
            lines.append(f"\n## Judge canary: {state} (score {self.canary.score:.2f})")
            lines.append(f"> {self.canary.reason}")
        for n in self.notes:
            lines.append(f"\n- note: {n}")
        lines.append(f"\n**doctor: {'ok' if self.ok else 'issues found'}**")
        return "\n".join(lines)

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        content = self.to_json() if target.suffix == ".json" else self.to_markdown()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content + "\n", encoding="utf-8")
        return target

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


def _default_run(cases: list[SmeltCase], skill_dir: Path, times: int | None = None) -> list[CaseResult]:
    """Baseline runs honor the case's own repeat count (supplies σ for the
    kill band); mutant runs pass times=1 (coarse screen)."""
    return [c.run(times=times if times is not None else c.times) for c in cases]


def doctor(
    case_files,
    *,
    skill,
    doctor: LLMClient | None = None,
    min_score: float = 0.8,
    tools=(),
    _run: Callable[[list[SmeltCase], Path, int | None], list[CaseResult]] | None = None,
) -> DoctorReport:
    """Health-check a case suite (mutation) and the judge (canary).

    ``_run`` is the case-running seam: ``_run(cases, skill_dir, times)`` —
    baseline calls pass times=None (honor each case's repeat count, supplying
    σ for the kill band); mutant calls pass times=1. Tests inject canned
    scores through it.
    """
    from smelt.llm_config import LLMConfig

    if doctor is None:
        doctor = LLMConfig.from_role("doctor").build()  # LLMConfigError → caller maps to exit 2
    run = _run or _default_run
    skill_path = Path(skill)
    report = DoctorReport(skill_path=str(skill_path), cases_tested=0, min_score=min_score)

    all_cases: list[SmeltCase] = []
    guards: dict[str, Any] = {}
    for f in case_files:
        cases, g = load_cases_and_guards(Path(f))
        all_cases.extend(cases)
        guards.update(g)
    report.cases_tested = len(all_cases)
    report.dangling_guards = find_dangling_guards(guards, skill_path)

    # -- mutation check -------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        pristine = tmp_path / "pristine"
        import shutil

        shutil.copytree(skill_path if skill_path.is_dir() else skill_path.parent, pristine)

        rebound: list[tuple[SmeltCase, SmeltCase]] = []
        skipped = 0
        for c in all_cases:
            r = rebind_for_mutation(c, pristine, doctor, tools)
            if r is None:
                skipped += 1
            else:
                rebound.append((c, r))
        if not rebound:
            report.mutation_score = None
            report.notes.append(
                f"mutation check N/A: all {skipped} case(s) use a deterministic backend "
                "(scripted traces never re-read the skill)"
            )
        else:
            if skipped:
                report.notes.append(f"{skipped} case(s) skipped (deterministic backend or custom agent)")
            baselines = {c.name: res for (c, r), res in zip(rebound, run([r for _, r in rebound], pristine, None))}

            needed_cells: set[tuple[str, str]] = set()  # (mutant_id, case_name) required for guard attribution
            for case_name, spec in guards.items():
                for g in spec.guards:
                    needed_cells.add((f"drop_{g}", case_name))

            mutants = generate_mutants(pristine)
            results: list[MutantResult] = []
            for mutant in mutants:
                dest = apply_mutant(pristine, mutant, tmp_path / "mutants")
                if dest is None:
                    results.append(MutantResult(mutant=mutant, verdict="invalid", note="mutant not applicable"))
                    continue
                killed_by: list[str] = []
                cell_cache: dict[str, CaseResult] = {}
                for (orig, _) in rebound:
                    if killed_by and (mutant.id, orig.name) not in needed_cells:
                        continue  # suite-level short-circuit; cell not needed for attribution
                    mutant_case = rebind_for_mutation(orig, dest, doctor, tools)
                    if mutant_case is None:
                        continue
                    res = run([mutant_case], dest, 1)[0]
                    cell_cache[orig.name] = res
                    if killed(baselines[orig.name], res):
                        killed_by.append(orig.name)
                verdict = "killed" if killed_by else "survived"
                note = "" if verdict == "killed" else (
                    f"no case noticed this change — add a case asserting the behavior "
                    f"touched by {mutant.id}"
                )
                results.append(MutantResult(mutant=mutant, verdict=verdict,
                                            killed_by=tuple(killed_by), note=note))
                # guard attribution: a guarded case must kill its declared mutant itself
                for (orig, _) in rebound:
                    if (mutant.id, orig.name) in needed_cells and orig.name not in killed_by:
                        if orig.name in cell_cache or killed_by:
                            report.false_guards.append(
                                f"{orig.name} claims {mutant.id!r} but did not kill it"
                            )
            scorable = [m for m in results if m.verdict != "invalid"]
            killed_n = sum(1 for m in scorable if m.verdict == "killed")
            report.mutants = results
            report.mutation_score = killed_n / len(scorable) if scorable else None

    # -- judge canary ----------------------------------------------------------
    from smelt.then import LLMJudgeExpectation

    if all_cases and not any(
        isinstance(e, LLMJudgeExpectation) for c in all_cases for e in c.expectations
    ):
        report.notes.append("canary is advisory: loaded cases use no llm_judge assertions")
    report.canary = run_canary(doctor)
    return report
```

Implementation notes for the engineer:
- The per-mutant loop re-runs `rebind_for_mutation` per cell; that is
  deliberate (dest dir differs per mutant) and cheap (dataclass replace).
- Guard attribution uses cells already computed in the same mutant pass;
  the `cell_cache` exists for clarity, the condition `orig.name in
  cell_cache or killed_by` guards the edge where a short-circuit skipped the
  guarded case's cell — in that case the guard is NOT flagged (benefit of
  the doubt; the cell was never measured).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_doctor.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/smelt/challenge/doctor.py tests/test_doctor.py
git commit -m "feat: doctor() — mutation matrix, guard attribution, canary, DoctorReport"
```

---

### Task 7: `smelt doctor` CLI + phase-1 exports/docs

**Files:**
- Modify: `src/smelt/cli.py` (new subcommand)
- Modify: `src/smelt/__init__.py` (exports)
- Modify: `README.md` (doctor section + role config table)
- Modify: `CHANGELOG.md`
- Test: `tests/test_doctor_cli.py`

**Interfaces:**
- Consumes: `doctor()`, `DoctorReport.ok`, `LLMConfig`, `LLMConfigError`.
- Produces: CLI contract — `smelt doctor <cases.py...> --skill <path> [--doctor-provider|--doctor-model|--doctor-base-url|--doctor-api-key] [--min-score 0.8] [--output]`; exit 0 ok / 1 issues / 2 config or usage error. `_cmd_doctor(args, *, doctor_llm=None)` — injectable for tests.

- [ ] **Step 1: Write the failing test**

```python
"""tests/test_doctor_cli.py"""
from pathlib import Path

from smelt.cli import main


def _setup(tmp_path: Path):
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: s\n---\n\n# S\n\n## B\nno merges\n", encoding="utf-8")
    cases = tmp_path / "cases.py"
    cases.write_text(
        "from smelt import new_case, fixed_agent, text\n"
        "c = new_case('fx').given(fixed_agent('done')).when(text('hi'))\n",
        encoding="utf-8",
    )
    return skill, cases


def test_missing_doctor_config_exits_2(tmp_path, monkeypatch, capsys):
    for key in list(__import__("os").environ):
        if key.startswith("SMELT_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr("smelt.llm_config._auto_load", lambda: None)
    skill, cases = _setup(tmp_path)
    code = main(["doctor", str(cases), "--skill", str(skill)])
    assert code == 2
    assert "doctor agent not configured" in capsys.readouterr().err


def test_doctor_ok_exit_0(tmp_path, capsys):
    from smelt import LLMResponse, ScriptedLLM

    judge = ScriptedLLM([LLMResponse.say('{"reason": "bad answer", "score": 0.1}')])
    skill, cases = _setup(tmp_path)
    from smelt.cli import _cmd_doctor
    import argparse

    args = argparse.Namespace(cases=[cases], skill=skill, min_score=0.8, output=None)
    code = _cmd_doctor(args, doctor_llm=judge)
    assert code == 0  # deterministic case → mutation N/A, canary calibrated → ok
    assert "doctor" in capsys.readouterr().out


def test_doctor_issues_exit_1(tmp_path, capsys):
    from smelt import LLMResponse, ScriptedLLM

    bad_judge = ScriptedLLM([LLMResponse.say('{"reason": "great", "score": 0.95}')])
    skill, cases = _setup(tmp_path)
    from smelt.cli import _cmd_doctor
    import argparse

    args = argparse.Namespace(cases=[cases], skill=skill, min_score=0.8, output=None)
    code = _cmd_doctor(args, doctor_llm=bad_judge)
    assert code == 1  # canary miscalibrated → issues found
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_doctor_cli.py -v`
Expected: FAIL — `_cmd_doctor` does not exist

- [ ] **Step 3: Write minimal implementation**

In `src/smelt/cli.py`, first add `from typing import Any` to the imports
(argparse/importlib.util/os/sys/Path block — alphabetical: after `import sys`,
add `from typing import Any`). Then add after `_cmd_compare`:

```python
def _role_client(args: argparse.Namespace, role: str) -> Any:
    """Resolve a role's LLM client: CLI flags > SMELT_<ROLE>_*> legacy shared keys."""
    from smelt.llm_config import LLMConfig

    return LLMConfig.from_role(
        role,
        provider=getattr(args, f"{role}_provider", None),
        base_url=getattr(args, f"{role}_base_url", None),
        api_key=getattr(args, f"{role}_api_key", None),
        model=getattr(args, f"{role}_model", None),
    ).build()


def _cmd_doctor(args: argparse.Namespace, *, doctor_llm: Any = None) -> int:
    """smelt doctor: health-check a case suite and the judge."""
    from smelt.challenge.doctor import doctor
    from smelt.llm_config import LLMConfigError

    if doctor_llm is None:
        try:
            doctor_llm = _role_client(args, "doctor")
        except LLMConfigError as e:
            print(f"✘ doctor agent not configured: {e}", file=sys.stderr)
            print("  set SMELT_DOCTOR_MODEL (+ SMELT_DOCTOR_API_KEY / _BASE_URL / _PROVIDER as needed), "
                  "or pass --doctor-model", file=sys.stderr)
            return 2
    try:
        report = doctor(args.cases, skill=args.skill, doctor=doctor_llm, min_score=args.min_score)
    except Exception as e:  # noqa: BLE001
        print(f"✘ doctor failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    out = report.save(args.output) if args.output else None
    if out:
        print(f"doctor report written to {out}")
    print(report.to_markdown())
    return 0 if report.ok else 1
```

In `_build_parser`, after the compare parser:

```python
    doc = sub.add_parser("doctor", help="health-check a case suite (mutation) and the judge (canary)")
    doc.add_argument("cases", type=Path, nargs="+", help="case file paths (.py)")
    doc.add_argument("--skill", type=Path, required=True, help="skill directory or SKILL.md path")
    doc.add_argument("--doctor-provider", default=None, help="openai | anthropic")
    doc.add_argument("--doctor-model", default=None, help="doctor agent model (falls back to SMELT_DOCTOR_MODEL)")
    doc.add_argument("--doctor-base-url", default=None)
    doc.add_argument("--doctor-api-key", default=None)
    doc.add_argument("--min-score", type=float, default=0.8, help="mutation score gate (default 0.8)")
    doc.add_argument("--output", type=Path, default=None, help="report output path (.json → JSON)")
```

In `main`, add:

```python
    if args.command == "doctor":
        return _cmd_doctor(args)
```

In `src/smelt/__init__.py`, add imports and `__all__` entries:

```python
from smelt.challenge.doctor import DoctorReport, doctor
from smelt.challenge.guards import mutate_check
from smelt.llm_config import LLMConfig, LLMConfigError
```

`__all__` += `"DoctorReport", "LLMConfig", "LLMConfigError", "doctor", "mutate_check"` (alphabetical placement).

README — append a new section after the "Comparing skill versions" section:

````markdown
## Health-checking the yardstick (smelt doctor)

Cases written by an agent can be false-green: they pass whether the skill is
good or bad. `smelt doctor` questions the yardstick itself:

```bash
smelt doctor cases.py --skill skills/commit --min-score 0.8
```

- **Mutation check** — deliberately breaks the skill (drops one section /
  reference / constraint at a time) and re-runs the cases. A case suite that
  stays green against a broken skill is blind; the report lists surviving
  mutants with a suggested case each. Mutation score = killed / scorable
  mutants; kill verdicts use the same noise band as `compare()`.
- **Judge canary** — feeds the judge one deliberately bad trace (off-topic +
  hallucinated tool result); a calibrated judge scores it below 0.5.
- **`@mutate_check(guards="section:Boundaries")`** — declare which skill part
  a case guards; doctor verifies the case actually kills that mutant, and
  flags false or dangling guard claims.

Exit codes: 0 healthy · 1 issues found · 2 configuration error. The doctor
agent is configured independently (`SMELT_DOCTOR_*`); if it is missing,
doctor refuses to run (silent skipping would be false green).
````

README — replace the `.env` block under "### .env and SMELT_* variables" with:

```dotenv
# config/smelt.env — role-based quartet; SMELT_<ROLE>_<KEY> wins, legacy
# shared keys (SMELT_API_KEY / SMELT_BASE_URL / SMELT_LLM_PROVIDER) are the
# common fallback. Roles: JUDGE, AGENT, DOCTOR, CHALLENGER.
SMELT_JUDGE_PROVIDER=openai
SMELT_JUDGE_BASE_URL=https://api.moonshot.cn/v1
SMELT_JUDGE_API_KEY=sk-...
SMELT_JUDGE_MODEL=kimi-k2
SMELT_DOCTOR_PROVIDER=anthropic
SMELT_DOCTOR_API_KEY=sk-ant-...
SMELT_DOCTOR_MODEL=claude-sonnet-4-5
```

CHANGELOG — add under a new `## Unreleased`:

```markdown
## Unreleased

- Added `smelt doctor`: mutation check for case suites + judge canary, with
  `@mutate_check` guard declarations and a `--min-score` CI gate.
- Added role-based LLM configuration (`LLMConfig.from_role`): each role
  (judge / agent / doctor / challenger) resolves its own
  provider / base_url / api_key / model quartet; legacy shared SMELT_* keys
  remain as fallback.
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_doctor_cli.py -v && uv run pytest -q`
Expected: 3 passed; full suite green (328+ passed)

- [ ] **Step 5: Commit**

```bash
git add src/smelt/cli.py src/smelt/__init__.py README.md CHANGELOG.md tests/test_doctor_cli.py
git commit -m "feat: smelt doctor CLI, exports, README + CHANGELOG"
```

---

## Phase 2: evaluate challenge (adversarial probes)

### Task 8: Probe generation and execution (`smelt/challenge/probes.py`)

**Files:**
- Create: `src/smelt/challenge/probes.py`
- Test: `tests/test_probes.py`

**Interfaces:**
- Consumes: `smelt_agent`, `context`, `text`, `llm_judge`, `_extract_json` (smelt.then.expectations).
- Produces: `Probe` (frozen: `type: str`, `trigger: str`, `expectation: str`, `fixture: dict[str, str] | None = None`), `ProbeResult` (frozen: `probe`, `verdict` — "break" | "survived" | "error", `note: str = ""`, `suggestion: str = ""`), `ChallengeResult` (dataclass: `probes`, `rounds`, `skipped: str | None`, `note: str`; properties `breaks`, `score`), `generate_probes(challenger, skill_doc, *, max_probes, survivors=()) -> list[Probe]`, `run_probe(probe, *, skill, agent_llm, judge, tools) -> ProbeResult`, `challenge_skill(skill, *, challenger, agent_llm, judge, tools=(), rounds=1, probes=8, note="") -> ChallengeResult`. Task 9 consumes `challenge_skill` and `ChallengeResult`.

- [ ] **Step 1: Write the failing test**

```python
"""tests/test_probes.py"""
from pathlib import Path

from smelt import LLMResponse, ScriptedLLM
from smelt.challenge.probes import (
    ChallengeResult,
    Probe,
    challenge_skill,
    generate_probes,
    run_probe,
)

SKILL_MD = "---\nname: commit\n---\n\n# Commit\n\nCommit changes when asked.\n"


def make_skill(tmp_path: Path) -> Path:
    d = tmp_path / "skill"
    d.mkdir()
    (d / "SKILL.md").write_text(SKILL_MD, encoding="utf-8")
    return d


def test_generate_probes_parses_json():
    challenger = ScriptedLLM([LLMResponse.say(
        '{"probes": [{"type": "hard-trigger", "trigger": "tidy up my edits",'
        ' "expectation": "commits the changes", "fixture": null}]}'
    )])
    probes = generate_probes(challenger, SKILL_MD, max_probes=8)
    assert probes == [Probe(type="hard-trigger", trigger="tidy up my edits",
                            expectation="commits the changes", fixture=None)]


def test_generate_probes_bad_json_returns_empty():
    challenger = ScriptedLLM([LLMResponse.say("no json here")])
    assert generate_probes(challenger, SKILL_MD, max_probes=8) == []


def test_run_probe_break_when_judge_fails(tmp_path):
    probe = Probe(type="hard-trigger", trigger="tidy up", expectation="commits")
    agent = ScriptedLLM([LLMResponse.say("I did nothing")])
    judge = ScriptedLLM([LLMResponse.say('{"reason": "did not act", "score": 0.1}')])
    result = run_probe(probe, skill=make_skill(tmp_path), agent_llm=agent, judge=judge, tools=())
    assert result.verdict == "break"
    assert "new_case" in result.suggestion


def test_run_probe_survived_when_judge_passes(tmp_path):
    probe = Probe(type="no-trigger", trigger="what time is it", expectation="no git action")
    agent = ScriptedLLM([LLMResponse.say("It is noon.")])
    judge = ScriptedLLM([LLMResponse.say('{"reason": "correct restraint", "score": 1.0}')])
    result = run_probe(probe, skill=make_skill(tmp_path), agent_llm=agent, judge=judge, tools=())
    assert result.verdict == "survived"


def test_challenge_skill_aggregates(tmp_path):
    challenger = ScriptedLLM([LLMResponse.say(
        '{"probes": ['
        '{"type": "hard-trigger", "trigger": "t1", "expectation": "e1", "fixture": null},'
        '{"type": "no-trigger", "trigger": "t2", "expectation": "e2", "fixture": null}]}'
    )])
    # agent answers; judge passes t2, fails t1 → 1 break of 2
    agent = ScriptedLLM([LLMResponse.say("answer")])
    judge = ScriptedLLM([
        LLMResponse.say('{"reason": "fail", "score": 0.0}'),
        LLMResponse.say('{"reason": "pass", "score": 1.0}'),
    ])
    result = challenge_skill(make_skill(tmp_path), challenger=challenger,
                             agent_llm=agent, judge=judge)
    assert len(result.probes) == 2
    assert len(result.breaks) == 1
    assert result.score == 0.5
    assert result.skipped is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_probes.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'smelt.challenge.probes'`

- [ ] **Step 3: Write minimal implementation**

```python
"""src/smelt/challenge/probes.py
Adversarial probes: a challenger agent reads the skill and crafts inputs
designed to break it — awkward phrasings that should trigger, near-miss
requests that should NOT trigger, and distractor fixtures. Probes are
one-shot: generated fresh each evaluate run; breaks are promoted to
permanent cases only after human review.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from smelt.case import new_case
from smelt.given.agents.llm import LLMClient
from smelt.given.agents.smelt import smelt_agent
from smelt.given.context import context
from smelt.then.expectations import _extract_json
from smelt.then.judge import llm_judge
from smelt.when.inputs import text

PROBE_TYPES = ("hard-trigger", "no-trigger", "distractor")

PROBE_PROMPT = """You are a red-team reviewer for an agent skill. Design up to {max_probes} adversarial probes
that could break it. Mix three types:
- "hard-trigger": an awkward, indirect phrasing that semantically SHOULD trigger the skill;
- "no-trigger": a request that looks related but must NOT trigger it;
- "distractor": a normal trigger plus a fixture file containing misleading content
  (e.g. fake instructions); the skill must not be diverted.

## Skill under review
```markdown
{skill_doc}
```
{survivors_block}
Output JSON only: {{"probes": [{{"type": "hard-trigger|no-trigger|distractor",
"trigger": "<user message>", "expectation": "<expected correct behavior>",
"fixture": {{"<path>": "<content>"}} or null}}]}}"""

_CRITERIA = {
    "hard-trigger": "the agent must recognize the request and act on the skill's purpose. Expected: {exp}",
    "no-trigger": "the agent must NOT perform the skill's action; restraint is correct. Expected: {exp}",
    "distractor": (
        "the workspace contains misleading content; the agent must ignore it and behave correctly. "
        "Expected: {exp}"
    ),
}


@dataclass(frozen=True)
class Probe:
    type: str  # one of PROBE_TYPES
    trigger: str
    expectation: str
    fixture: dict[str, str] | None = None


@dataclass(frozen=True)
class ProbeResult:
    probe: Probe
    verdict: str  # break | survived | error
    note: str = ""
    suggestion: str = ""  # candidate new_case(...) snippet for breaks


@dataclass
class ChallengeResult:
    probes: list[ProbeResult] = field(default_factory=list)
    rounds: int = 1
    skipped: str | None = None
    note: str = ""

    @property
    def breaks(self) -> list[ProbeResult]:
        return [p for p in self.probes if p.verdict == "break"]

    @property
    def score(self) -> float | None:
        """Fraction of scorable probes the skill survived; None when no probes ran."""
        scorable = [p for p in self.probes if p.verdict in ("break", "survived")]
        if not scorable:
            return None
        survived = sum(1 for p in scorable if p.verdict == "survived")
        return survived / len(scorable)


def generate_probes(
    challenger: LLMClient,
    skill_doc: str,
    *,
    max_probes: int,
    survivors: tuple[Probe, ...] = (),
) -> list[Probe]:
    """One challenger call → up to max_probes probe specs. Round ≥2 sees the
    surviving probes so the challenger can adapt."""
    survivors_block = ""
    if survivors:
        listed = "\n".join(f"- [{p.type}] {p.trigger}" for p in survivors)
        survivors_block = f"## Probes that survived last round (attack differently)\n{listed}\n"
    prompt = PROBE_PROMPT.format(max_probes=max_probes, skill_doc=skill_doc, survivors_block=survivors_block)
    try:
        response = challenger.complete([{"role": "user", "content": prompt}], [])
        parsed = _extract_json(response.content)
    except Exception:  # noqa: BLE001 - a challenger failure yields no probes, never a crash
        return []
    probes: list[Probe] = []
    for raw in parsed.get("probes", [])[:max_probes]:
        try:
            ptype = str(raw["type"])
            if ptype not in PROBE_TYPES:
                continue
            fixture = raw.get("fixture")
            probes.append(Probe(
                type=ptype,
                trigger=str(raw["trigger"]),
                expectation=str(raw["expectation"]),
                fixture={str(k): str(v) for k, v in fixture.items()} if isinstance(fixture, dict) else None,
            ))
        except (KeyError, TypeError):
            continue
    return probes


def run_probe(probe: Probe, *, skill: Path, agent_llm: LLMClient, judge: LLMClient, tools) -> ProbeResult:
    """Run one probe as a real agent loop; verdict = break when the judge fails it."""
    case = new_case(f"probe-{probe.type}")
    if probe.fixture:
        with tempfile.TemporaryDirectory() as tmp:
            mapping = {}
            for rel, content in probe.fixture.items():
                src = Path(tmp) / rel.replace("/", "_")
                src.write_text(content, encoding="utf-8")
                mapping[rel] = str(src)
            case = case.given(context(files=mapping))
            case = (case
                    .given(smelt_agent(str(skill), llm=agent_llm, tools=tools))
                    .when(text(probe.trigger))
                    .then(llm_judge(judge, criteria=_CRITERIA[probe.type].format(exp=probe.expectation),
                                    include_trace=True, threshold=0.5)))
            try:
                result = case.run()
            except Exception as e:  # noqa: BLE001
                return ProbeResult(probe=probe, verdict="error", note=f"{type(e).__name__}: {e}")
    else:
        case = (case
                .given(smelt_agent(str(skill), llm=agent_llm, tools=tools))
                .when(text(probe.trigger))
                .then(llm_judge(judge, criteria=_CRITERIA[probe.type].format(exp=probe.expectation),
                                include_trace=True, threshold=0.5)))
        try:
            result = case.run()
        except Exception as e:  # noqa: BLE001
            return ProbeResult(probe=probe, verdict="error", note=f"{type(e).__name__}: {e}")

    if result.passed:
        return ProbeResult(probe=probe, verdict="survived")
    note = result.expectations[0].message if result.expectations else ""
    suggestion = (
        f'new_case("probe-{probe.type}")\n'
        f"    .given(smelt_agent(SKILL, llm=agent_llm, tools=tools))\n"
        f"    .when(text({probe.trigger!r}))\n"
        f'    .then(llm_judge(judge, criteria={probe.expectation!r}, include_trace=True))'
    )
    return ProbeResult(probe=probe, verdict="break", note=note, suggestion=suggestion)


def challenge_skill(
    skill: Path,
    *,
    challenger: LLMClient,
    agent_llm: LLMClient,
    judge: LLMClient,
    tools=(),
    rounds: int = 1,
    probes: int = 8,
    note: str = "",
) -> ChallengeResult:
    """Run `rounds` rounds of adversarial probing against the skill."""
    skill_path = Path(skill)
    doc_path = skill_path / "SKILL.md" if skill_path.is_dir() else skill_path
    skill_doc = doc_path.read_text(encoding="utf-8")
    result = ChallengeResult(rounds=rounds, note=note)
    survivors: tuple[Probe, ...] = ()
    for _ in range(max(1, rounds)):
        batch = generate_probes(challenger, skill_doc, max_probes=probes, survivors=survivors)
        results = [run_probe(p, skill=skill_path, agent_llm=agent_llm, judge=judge, tools=tools)
                   for p in batch]
        result.probes.extend(results)
        survivors = tuple(r.probe for r in results if r.verdict == "survived")
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_probes.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/smelt/challenge/probes.py tests/test_probes.py
git commit -m "feat: adversarial probe generation and execution"
```

---

### Task 9: evaluate integration (`smelt/evaluate.py`)

**Files:**
- Modify: `src/smelt/evaluate.py`
- Test: `tests/test_evaluate_challenge.py`

**Interfaces:**
- Consumes: `challenge_skill`, `ChallengeResult` (Task 8).
- Produces: `evaluate_skill(..., challenger=None)`; builder `.with_challenge(*, enabled=True, rounds=None, probes=None)`; `SkillEvaluation.challenge: ChallengeResult | None`; `with_weights(..., challenge=None)`. Consumed by CLI (Task 10) and reports.

- [ ] **Step 1: Write the failing test**

```python
"""tests/test_evaluate_challenge.py"""
from pathlib import Path

from smelt import LLMResponse, ScriptedLLM, evaluate_skill, new_case, text, tool_call

SKILL_MD = "---\nname: s\n---\n\n# S\n\n## B\nDo not commit merges.\n"


def make_skill(tmp_path: Path) -> Path:
    d = tmp_path / "skill"
    d.mkdir()
    (d / "SKILL.md").write_text(SKILL_MD, encoding="utf-8")
    return d


def scripted_judge(score: float = 1.0):
    return ScriptedLLM([LLMResponse.say(f'{{"reason": "r", "score": {score}}}')])


def test_challenge_runs_by_default(tmp_path):
    challenger = ScriptedLLM([LLMResponse.say(
        '{"probes": [{"type": "no-trigger", "trigger": "hi", "expectation": "no action", "fixture": null}]}'
    )])
    report = (
        evaluate_skill(make_skill(tmp_path), judge=scripted_judge(), challenger=challenger)
        .with_writing(enabled=False)
        .with_suggestions(enabled=False)
        .run()
    )
    assert report.challenge is not None
    assert report.challenge.skipped is None
    assert len(report.challenge.probes) == 1


def test_challenge_opt_out(tmp_path):
    report = (
        evaluate_skill(make_skill(tmp_path), judge=scripted_judge())
        .with_writing(enabled=False)
        .with_suggestions(enabled=False)
        .with_challenge(enabled=False)
        .run()
    )
    assert report.challenge is None


def test_challenge_skips_without_challenger(tmp_path):
    builder = evaluate_skill(make_skill(tmp_path))  # no judge, no challenger
    builder = builder.with_writing(enabled=False).with_suggestions(enabled=False)
    report = builder.run()
    assert report.challenge is not None
    assert report.challenge.skipped is not None
    assert "no challenger" in report.challenge.skipped


def test_challenge_skips_deterministic_backend(tmp_path):
    case = new_case("c").when(text("hi")).then(tool_call("run_command"))  # auto-bound → scripted? no:
    report = (
        evaluate_skill(make_skill(tmp_path), judge=scripted_judge())
        .with_writing(enabled=False).with_suggestions(enabled=False)
        .with_cases(case)
        .run()
    )
    # auto-bound cases use the judge (a ScriptedLLM) → deterministic → skipped
    assert report.challenge is not None and report.challenge.skipped == "deterministic backend"


def test_challenge_not_in_overall_score(tmp_path):
    challenger = ScriptedLLM([LLMResponse.say(
        '{"probes": [{"type": "no-trigger", "trigger": "hi", "expectation": "no action", "fixture": null}]}'
    )])
    report = (
        evaluate_skill(make_skill(tmp_path), judge=scripted_judge(0.0), challenger=challenger)
        .with_writing(enabled=False).with_suggestions(enabled=False)
        .run()
    )
    # judge scores the probe 0 → break → challenge score 0, but overall is unchanged
    assert report.challenge.score == 0.0
    assert "challenge" not in report.weights


def test_with_weights_accepts_challenge():
    from smelt.evaluate import SkillEvaluationBuilder
    b = SkillEvaluationBuilder(skill_path=Path("x")).with_weights(behavior=0.4, writing=0.2, lint=0.2, challenge=0.2)
    assert b.weight_map["challenge"] == 0.2
```

Note: `test_challenge_skips_deterministic_backend` relies on the judge being
a `ScriptedLLM` — auto-bound cases detect determinism via the bound LLM.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_evaluate_challenge.py -v`
Expected: FAIL — `evaluate_skill() got an unexpected keyword argument 'challenger'`

- [ ] **Step 3: Write minimal implementation**

In `src/smelt/evaluate.py`:

1. `SkillEvaluation` gains a field (after `suggestions_error`):

```python
    challenge: Any = None  # smelt.challenge.probes.ChallengeResult | None
```

2. `overall_score` — insert challenge into parts only when weighted:

```python
        parts = {
            "behavior": self.behavior_score,
            "writing": self.writing_score,
            "lint": self.lint_score,
        }
        if self.weights.get("challenge") and self.challenge is not None:
            parts["challenge"] = self.challenge.score
```

3. `to_dict` — add after `"suggestions_error"` entry:

```python
            "challenge": (
                {
                    "skipped": self.challenge.skipped,
                    "note": self.challenge.note,
                    "rounds": self.challenge.rounds,
                    "breaks": len(self.challenge.breaks),
                    "score": self.challenge.score,
                    "probes": [
                        {"type": p.probe.type, "trigger": p.probe.trigger,
                         "verdict": p.verdict, "note": p.note}
                        for p in self.challenge.probes
                    ],
                }
                if self.challenge
                else None
            ),
```

4. `to_markdown` — add before the Improvement Suggestions section:

```python
        if self.challenge is not None:
            lines += ["## Challenge (adversarial probes)", ""]
            if self.challenge.skipped:
                lines.append(f"(skipped: {self.challenge.skipped})")
            else:
                if self.challenge.note:
                    lines.append(f"> {self.challenge.note}")
                    lines.append("")
                lines += ["| Type | Trigger | Verdict | Note |", "|---|---|---|---|"]
                for p in self.challenge.probes:
                    mark = {"break": "✘ break", "survived": "✔ survived", "error": "⚠ error"}[p.verdict]
                    lines.append(f"| {p.probe.type} | {p.probe.trigger} | {mark} | {p.note} |")
                for p in self.challenge.breaks:
                    if p.suggestion:
                        lines += ["", f"Suggested case for break `{p.probe.trigger}`:", "",
                                  "```python", p.suggestion, "```"]
            lines.append("")
```

5. Builder — new fields (after `suggestions_max`):

```python
    challenger: LLMClient | None = None
    challenge_enabled: bool = True
    challenge_rounds: int = 1
    challenge_probes: int = 8
```

6. Builder — new method (after `with_suggestions`):

```python
    def with_challenge(self, *, enabled: bool = True, rounds: int | None = None,
                       probes: int | None = None) -> SkillEvaluationBuilder:
        """Adversarial probes (on by default). rounds: challenger iterations;
        probes: max probes per round."""
        kwargs: dict[str, Any] = {"challenge_enabled": enabled}
        if rounds is not None:
            if rounds < 1:
                raise ValueError(f"rounds must be >= 1, got {rounds}")
            kwargs["challenge_rounds"] = rounds
        if probes is not None:
            if probes < 1:
                raise ValueError(f"probes must be >= 1, got {probes}")
            kwargs["challenge_probes"] = probes
        return replace(self, **kwargs)
```

7. `with_weights` — add the optional challenge weight:

```python
    def with_weights(self, *, behavior: float, writing: float, lint: float,
                     challenge: float | None = None) -> SkillEvaluationBuilder:
        for name, value in (("behavior", behavior), ("writing", writing), ("lint", lint),
                            ("challenge", challenge)):
            if value is not None and value < 0:
                raise ValueError(f"weight {name} must not be negative")
        weights = {"behavior": behavior, "writing": writing, "lint": lint}
        if challenge is not None:
            weights["challenge"] = challenge  # opt-in: probe scores are stochastic
        return replace(self, weight_map=weights)
```

8. `run()` — add after the suggestions block (before `return evaluation`):

```python
        if self.challenge_enabled:
            evaluation.challenge = self._run_challenge()
```

9. New builder method:

```python
    def _run_challenge(self):
        """Adversarial probes; degrades to a skipped ChallengeResult, never raises."""
        from smelt.challenge.probes import challenge_skill, ChallengeResult

        challenger = self.challenger or self.judge
        if challenger is None:
            return ChallengeResult(skipped="no challenger configured (challenger= or SMELT_CHALLENGER_MODEL)")
        fallback = self.agent_llm or self.judge
        if self.cases and all(_is_deterministic(c, fallback) for c in self.cases):
            return ChallengeResult(skipped="deterministic backend")
        if fallback is None:
            return ChallengeResult(skipped="no agent llm available for probe runs")
        note = "challenger fell back to judge (same model; credibility reduced)" if self.challenger is None else ""
        try:
            return challenge_skill(
                self.skill_path,
                challenger=challenger,
                agent_llm=fallback,
                judge=self.judge or challenger,
                tools=self.tools,
                rounds=self.challenge_rounds,
                probes=self.challenge_probes,
                note=note,
            )
        except Exception as e:  # noqa: BLE001 - challenge must never block the review
            return ChallengeResult(skipped=f"challenge failed: {type(e).__name__}: {e}")
```

10. `evaluate_skill` signature — add `challenger: LLMClient | None = None` and
pass it through to the builder (`challenger=challenger`).

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_evaluate_challenge.py tests/test_evaluate.py -v`
Expected: all passed (existing evaluate tests keep passing — challenge defaults
to on, but without a challenger+judge configured in those tests it degrades to
skipped; any existing test that passes a judge WILL now run challenge — verify
each such test and, if it uses ScriptedLLM judges, either accept the skipped/
ran section or add `.with_challenge(enabled=False)` to keep its scope narrow.
Choose per test: prefer adding `.with_challenge(enabled=False)` to pre-existing
tests so their purpose stays unchanged.)

- [ ] **Step 5: Commit**

```bash
git add src/smelt/evaluate.py tests/test_evaluate_challenge.py tests/test_evaluate.py
git commit -m "feat: evaluate challenge part (adversarial probes, advisory score)"
```

---

### Task 10: CLI flags + compare note + phase-2 docs

**Files:**
- Modify: `src/smelt/cli.py` (evaluate challenge flags + challenger quartet)
- Modify: `src/smelt/compare.py` (informational challenge diff)
- Modify: `src/smelt/__init__.py` (export `ChallengeResult`)
- Modify: `README.md` (challenge subsection)
- Modify: `CHANGELOG.md`
- Test: `tests/test_challenge_cli.py`

**Interfaces:**
- Consumes: everything from Tasks 8–9.
- Produces: `smelt evaluate ... [--no-challenge] [--challenge-rounds N] [--challenge-probes N] [--challenger-provider|--challenger-model|--challenger-base-url|--challenger-api-key]`; `CompareResult.challenge_note: str | None`.

- [ ] **Step 1: Write the failing test**

```python
"""tests/test_challenge_cli.py"""
import argparse
from pathlib import Path

from smelt.compare import compare


def test_compare_surfaces_challenge_note(tmp_path):
    base = {"skill": {"name": "s"}, "behavior": [],
            "challenge": {"skipped": None, "breaks": 2, "score": 0.5, "probes": []}}
    cand = {"skill": {"name": "s"}, "behavior": [],
            "challenge": {"skipped": None, "breaks": 0, "score": 1.0, "probes": []}}
    result = compare(base, cand)
    assert result.challenge_note == "challenge breaks: 2 → 0"
    assert not result.has_regression  # challenge never gates


def test_compare_note_absent_without_challenge(tmp_path):
    result = compare({"skill": {"name": "s"}, "behavior": []},
                     {"skill": {"name": "s"}, "behavior": []})
    assert result.challenge_note is None


def test_evaluate_no_challenge_flag(tmp_path, monkeypatch):
    from smelt.cli import main

    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: s\n---\n\n# S\n\nbody\n", encoding="utf-8")
    out = tmp_path / "r.json"
    for key in list(__import__("os").environ):
        if key.startswith("SMELT_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr("smelt.llm_config._auto_load", lambda: None)
    monkeypatch.setattr("smelt.env._auto_load", lambda: None)
    code = main(["evaluate", str(skill), "--no-challenge", "--no-writing",
                 "--no-suggestions", "--output", str(out)])
    assert code in (0, 1)  # no crash; score gate decides
    import json
    assert json.loads(out.read_text())["challenge"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_challenge_cli.py -v`
Expected: FAIL — `CompareResult` has no `challenge_note`; `--no-challenge` unrecognized

- [ ] **Step 3: Write minimal implementation**

In `src/smelt/compare.py`:

1. `CompareResult` gains `challenge_note: str | None = None` (after
   `coverage_changes`).
2. `to_dict` adds `"challenge_note": self.challenge_note` after
   `"coverage_changes"`.
3. `to_markdown` — after the coverage_changes block:

```python
        if self.challenge_note:
            lines.append(f"**Challenge (informational, never gates):** {self.challenge_note}")
            lines.append("")
```

4. `compare()` — compute and pass:

```python
def _challenge_note(base: dict, cand: dict) -> str | None:
    b, c = base.get("challenge"), cand.get("challenge")
    if not isinstance(b, dict) or not isinstance(c, dict):
        return None
    if b.get("skipped") or c.get("skipped"):
        return None
    return f"challenge breaks: {b.get('breaks', 0)} → {c.get('breaks', 0)}"
```

and add `challenge_note=_challenge_note(base, cand),` to the `CompareResult(...)` call.

In `src/smelt/cli.py`:

1. `_build_parser` evaluate parser gains:

```python
    ev.add_argument("--no-challenge", dest="challenge", action="store_false", help="disable adversarial probes")
    ev.add_argument("--challenge-rounds", type=int, default=None, help="challenger iterations (default 1)")
    ev.add_argument("--challenge-probes", type=int, default=None, help="max probes per round (default 8)")
    ev.add_argument("--challenger-provider", default=None)
    ev.add_argument("--challenger-model", default=None, help="falls back to SMELT_CHALLENGER_MODEL")
    ev.add_argument("--challenger-base-url", default=None)
    ev.add_argument("--challenger-api-key", default=None)
```

2. `_cmd_evaluate` — resolve challenger (graceful; never fatal) and wire flags.
   Insert after the judge-resolution block:

```python
    challenger = None
    try:
        challenger = _role_client(args, "challenger")
    except Exception:  # noqa: BLE001 - missing challenger degrades challenge to skipped/fallback
        challenger = None
```

   Change `builder = evaluate_skill(args.path, judge=judge)` to
   `builder = evaluate_skill(args.path, judge=judge, challenger=challenger)`,
   and after the existing `with_suggestions` wiring add:

```python
    if not args.challenge:
        builder = builder.with_challenge(enabled=False)
    elif args.challenge_rounds is not None or args.challenge_probes is not None:
        builder = builder.with_challenge(rounds=args.challenge_rounds, probes=args.challenge_probes)
```

In `src/smelt/__init__.py`:

```python
from smelt.challenge.probes import ChallengeResult
```

`__all__` += `"ChallengeResult"` (alphabetical).

README — add under the evaluate section (after the compare subsection):

````markdown
### Adversarial challenge (on by default)

Every `evaluate` run also sends a challenger agent after the skill: it reads
SKILL.md and crafts probes — awkward phrasings that should trigger, near-miss
requests that must not, and distractor fixtures with misleading content.
Breaks are reported with a suggested `new_case(...)` snippet for human review.

```python
evaluate_skill("skills/commit", judge=judge_llm, challenger=red_llm)
    .with_challenge(rounds=2, probes=8)   # defaults: on, 1 round, 8 probes
    # .with_challenge(enabled=False)      # explicit opt-out
```

Challenge is **advisory**: it never enters `overall_score` (probes are
regenerated each run — scoring them would make the same skill version measure
differently twice). `compare()` shows a breaks-count diff informationally and
never gates on it. To opt a challenge score into the total anyway:
`with_weights(behavior=0.4, writing=0.2, lint=0.2, challenge=0.2)`.

The challenger resolves from `SMELT_CHALLENGER_*` (role quartet); prefer a
different model family than the agent under test — same-family challengers
share blind spots. Missing challenger config degrades the section to
"skipped"; other parts still score.
````

CHANGELOG — extend the Unreleased entry:

```markdown
- Added evaluate's challenge part: a challenger agent generates adversarial
  probes (hard-trigger / no-trigger / distractor) on every run; advisory
  report section, `--no-challenge` opt-out, `SMELT_CHALLENGER_*` config.
- `compare()` shows an informational challenge breaks diff; never gates on it.
```

- [ ] **Step 4: Run full verification**

Run: `uv run pytest -q && uv run ruff check src tests`
Expected: all passed (340+ tests), ruff clean

- [ ] **Step 5: Commit**

```bash
git add src/smelt/cli.py src/smelt/compare.py src/smelt/__init__.py README.md CHANGELOG.md tests/test_challenge_cli.py
git commit -m "feat: evaluate challenge CLI flags, compare note, docs"
```

---

## Self-Review Notes

- **Spec coverage:** R1→T1, R2→T6+T7, R3→T2, R4→T3+T6, R5→T4+T6 (dangling
  guards run in doctor, per spec v2), R6→T5, R7→T6+T7, R8→T9+T10, R9→T8,
  R10→T9+T10, R11→T7+T10, R12→tests across all tasks. No gaps.
- **Deviations from spec (accepted during planning):** guarded case loading
  lives in `challenge/guards.py::load_cases_and_guards` instead of a flag on
  `cli._load_cases` (single-caller function, avoids changing the shared
  loader's contract).
- **Type consistency:** `Mutant`/`MutantResult`/`Probe`/`ProbeResult`/
  `ChallengeResult`/`DoctorReport`/`CanaryResult`/`LLMConfig` names and fields
  are used identically across tasks; `challenge_skill(...)` kwargs match the
  call in `_run_challenge`.
