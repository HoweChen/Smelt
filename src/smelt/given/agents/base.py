"""Agent protocol: the uniform interface for all testable agents (LLM-driven or fixed-output)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from smelt.given.context import CaseContext
from smelt.trace import Trace
from smelt.when.inputs import CaseInput


@runtime_checkable
class Agent(Protocol):
    """Given a context and a trigger, produce a Trace.

    Testers can implement their own Agent (wrapping a real CLI, SDK, or
    subprocess) — anything returning a smelt.trace.Trace plugs into the whole
    family of then-assertions.
    """

    def run(self, ctx: CaseContext, input: CaseInput) -> Trace: ...
