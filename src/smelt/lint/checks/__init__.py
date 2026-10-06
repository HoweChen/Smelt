"""Check registry. Register new checks in this file."""

from __future__ import annotations

from smelt.lint.checks.assets import AssetCheck
from smelt.lint.checks.base import Check
from smelt.lint.checks.clarity import ClarityCheck
from smelt.lint.checks.metadata import MetadataCheck
from smelt.lint.checks.structure import StructureCheck
from smelt.lint.checks.testcoverage import TestCoverageCheck
from smelt.lint.checks.trigger import TriggerCheck
from smelt.lint.models import CheckResult, SkillDoc

ALL_CHECKS: list[Check] = [
    MetadataCheck(),
    StructureCheck(),
    TriggerCheck(),
    ClarityCheck(),
    AssetCheck(),
    TestCoverageCheck(),
]


def run_checks(skill: SkillDoc, checks: list[Check] | None = None) -> list[CheckResult]:
    """Run all (or the given) checks against one skill document."""
    active = checks if checks is not None else ALL_CHECKS
    return [c.run(skill) for c in active]
