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
