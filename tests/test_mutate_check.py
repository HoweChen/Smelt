"""tests/test_mutate_check.py"""
from pathlib import Path

from smelt.challenge.guards import MutateSpec, find_dangling_guards, load_cases_and_guards, mutate_check

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
    skill = make_skill(tmp_path)
    guards = {"c": MutateSpec(guards=("section:Nope", "reference:references/gone.md"))}
    dangling = find_dangling_guards(guards, skill)
    assert any("'Nope'" in d for d in dangling)
    assert any("references/gone.md" in d for d in dangling)


def test_valid_guards_not_dangling(tmp_path):
    skill = make_skill(tmp_path)
    guards = {"c": MutateSpec(guards=("section:Boundaries",))}
    assert find_dangling_guards(guards, skill) == []
