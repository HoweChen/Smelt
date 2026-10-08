"""tests/test_mutation_kill.py"""

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
