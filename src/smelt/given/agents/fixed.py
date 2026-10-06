"""FixedAgent: a fixed-output agent with no LLM involved — for replay and assertion self-tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from smelt.given.context import CaseContext
from smelt.trace import ToolCallRecord, Trace
from smelt.when.inputs import CaseInput


@dataclass
class FixedAgent:
    """Replay a preconfigured trace: fixed tool-call sequence + fixed final output.

    Good for: 1) verifying then-assertions themselves; 2) recording a real run's
    trace into a regression case.
    """

    output: str = ""
    tool_calls: Sequence[Mapping[str, Any]] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def run(self, ctx: CaseContext, input: CaseInput) -> Trace:
        records = [
            ToolCallRecord(name=c["name"], arguments=dict(c.get("arguments", {})), result=c.get("result"))
            for c in self.tool_calls
        ]
        return Trace(output=self.output, tool_calls=list(records), metadata=dict(self.metadata), turns=1)


def fixed_agent(
    output: str = "",
    *,
    tool_calls: Sequence[Mapping[str, Any]] = (),
) -> FixedAgent:
    """given(fixed_agent('{"ok": true}', tool_calls=[{"name": "run_command", "arguments": {"cmd": "ls"}}]))"""
    return FixedAgent(output=output, tool_calls=tool_calls)
