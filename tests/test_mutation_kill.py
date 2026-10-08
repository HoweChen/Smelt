"""tests/test_mutation_kill.py"""

import shutil
from pathlib import Path

from smelt import (
    LLMResponse,
    ScriptedLLM,
    context,
    fixed_agent,
    new_case,
    reference,
    reference_folder,
    smelt_agent,
    text,
    tool_call,
)
from smelt.challenge.mutation import killed, rebind_for_mutation
from smelt.given.context import CaseContext
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


# -- context-mount rebinding ---------------------------------------------------
# A case that mounts the skill's resources via reference()/reference_folder()
# must perceive drop_reference mutants: mounts pointing under the original
# skill root are re-pointed at the (possibly mutated) copy.


def _skill_tree(root: Path) -> Path:
    d = root / "skills" / "demo"
    (d / "references").mkdir(parents=True)
    (d / "SKILL.md").write_text("---\nname: demo\ndescription: x\n---\nbody", encoding="utf-8")
    (d / "references" / "a.md").write_text("alpha", encoding="utf-8")
    (d / "references" / "b.md").write_text("beta", encoding="utf-8")
    return d


def _materialize(case, ws: Path) -> CaseContext:
    ws.mkdir(parents=True, exist_ok=True)
    ctx = CaseContext(workspace=ws)
    ctx.materialize(list(case.contexts))
    return ctx


def test_rebind_repoints_reference_folder_to_mutant_copy(tmp_path):
    skill = _skill_tree(tmp_path)
    mutant = tmp_path / "mutant"
    shutil.copytree(skill, mutant)
    (mutant / "references" / "a.md").unlink()  # drop_reference:references/a.md

    case = new_case("x").given(reference_folder(skill)).when(text("hi"))
    rebound = rebind_for_mutation(case, mutant, None, (), source_root=skill)
    assert rebound is not None

    ctx = _materialize(rebound, tmp_path / "ws")
    assert (ctx.workspace / "references" / "b.md").exists()
    assert not (ctx.workspace / "references" / "a.md").exists()  # mutation perceivable


def test_rebind_drops_single_file_mount_removed_by_mutant(tmp_path):
    skill = _skill_tree(tmp_path)
    mutant = tmp_path / "mutant"
    shutil.copytree(skill, mutant)
    (mutant / "references" / "a.md").unlink()

    case = new_case("x").given(reference(skill / "references" / "a.md")).when(text("hi"))
    rebound = rebind_for_mutation(case, mutant, None, (), source_root=skill)
    assert rebound is not None

    ctx = _materialize(rebound, tmp_path / "ws")
    assert not (ctx.workspace / "references" / "a.md").exists()


def test_rebind_repoints_pristine_baseline_too(tmp_path):
    # Baseline runs rebind to the pristine copy: all mounts exist there.
    skill = _skill_tree(tmp_path)
    pristine = tmp_path / "pristine"
    shutil.copytree(skill, pristine)

    case = new_case("x").given(reference_folder(skill)).when(text("hi"))
    rebound = rebind_for_mutation(case, pristine, None, (), source_root=skill)
    assert rebound is not None

    ctx = _materialize(rebound, tmp_path / "ws")
    assert (ctx.workspace / "references" / "a.md").exists()
    assert ctx.references == ["references/a.md", "references/b.md"]


def test_rebind_keeps_mounts_outside_skill_root(tmp_path):
    skill = _skill_tree(tmp_path)
    fixture = tmp_path / "fixtures" / "input.csv"
    fixture.parent.mkdir()
    fixture.write_text("1,2", encoding="utf-8")

    case = new_case("x").given(context(files={"data/input.csv": fixture})).when(text("hi"))
    rebound = rebind_for_mutation(case, skill, None, (), source_root=skill)
    assert rebound is not None
    assert [src for spec in rebound.contexts for src, _ in spec.files] == [fixture]
