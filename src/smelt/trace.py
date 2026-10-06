"""Execution trace: every piece of assertable evidence an agent leaves behind."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolCallRecord:
    """Record of a single tool call."""

    name: str
    arguments: dict[str, Any]
    result: Any = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class Trace:
    """Complete trace of one case run.

    - ``output``: the agent's final text output (usually the user-facing answer).
    - ``tool_calls``: tool calls made during the run, in order.
    - ``messages``: LLM conversation history (if any), for debugging.
    - ``metadata``: arbitrary agent-attached info (model name, latency, ...).
    """

    output: str = ""
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def calls_named(self, name: str) -> list[ToolCallRecord]:
        """All tool calls named ``name``."""
        return [c for c in self.tool_calls if c.name == name]

    @property
    def called_tools(self) -> list[str]:
        return [c.name for c in self.tool_calls]
