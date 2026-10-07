"""Reference assertions: reach (read when needed) and restraint (not read when
unneeded). Tool-name-agnostic: any string argument of any tool call counts."""

from __future__ import annotations

from dataclasses import dataclass

from smelt.refpath import calls_reading
from smelt.results import ExpectationResult
from smelt.then.expectations import _result
from smelt.trace import Trace


@dataclass(frozen=True)
class ReferenceReadExpectation:
    path: str
    threshold: float = 1.0

    @property
    def name(self) -> str:
        return f"reference_read({self.path})"

    def evaluate(self, trace: Trace) -> ExpectationResult:
        hits = calls_reading(trace, self.path)
        if hits:
            return _result(self.name, 1.0, self.threshold, f"read {len(hits)} time(s)")
        return _result(self.name, 0.0, self.threshold,
                       f"never read; actual calls: {trace.called_tools or '(none)'}")


@dataclass(frozen=True)
class NoReferenceReadExpectation:
    path: str
    threshold: float = 1.0

    @property
    def name(self) -> str:
        return f"no_reference_read({self.path})"

    def evaluate(self, trace: Trace) -> ExpectationResult:
        hits = calls_reading(trace, self.path)
        if hits:
            return _result(self.name, 0.0, self.threshold, f"read {len(hits)} time(s)")
        return _result(self.name, 1.0, self.threshold)


def reference_read(path: str, *, threshold: float = 1.0) -> ReferenceReadExpectation:
    """then(reference_read("references/endpoints.md")) — the reference was consulted."""
    return ReferenceReadExpectation(path=path, threshold=threshold)


def no_reference_read(path: str, *, threshold: float = 1.0) -> NoReferenceReadExpectation:
    """then(no_reference_read("references/performance.md")) — restraint."""
    return NoReferenceReadExpectation(path=path, threshold=threshold)
