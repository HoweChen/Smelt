"""Reference assertions: reach (read when needed), restraint (not read when
unneeded), and ablation validity (content never entered the context).
Tool-name-agnostic: any string argument of any tool call counts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

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


def _fingerprint_lines(file: Path, *, max_lines: int = 5, min_len: int = 20) -> list[str]:
    """Up to max_lines distinctive content lines (>= min_len chars), spread
    across the file — the proof of 'content entered the context'."""
    lines = [ln.strip() for ln in file.read_text(encoding="utf-8").splitlines()]
    lines = [ln for ln in lines if len(ln) >= min_len]
    if len(lines) <= max_lines:
        return lines
    step = len(lines) / max_lines
    return [lines[int(i * step)] for i in range(max_lines)]


@dataclass(frozen=True)
class ReferenceUntouchedExpectation:
    """Ablation validity: the reference's content never entered the context.

    Checks calls (path match), then tool RESULTS (fingerprint leak via detour
    tools), then the final output (parametric-memory contamination)."""

    path: str
    source: Path
    threshold: float = 1.0

    @property
    def name(self) -> str:
        return f"reference_untouched({self.path})"

    def evaluate(self, trace: Trace) -> ExpectationResult:
        calls = calls_reading(trace, self.path)
        if calls:
            return _result(self.name, 0.0, self.threshold,
                           f"read via {calls[0].name}({calls[0].arguments})")
        real = Path(self.source)
        if real.is_dir():
            real = real / self.path
        if not real.exists():
            return _result(self.name, 0.0, self.threshold, f"source not found: {real}")
        fingerprint = _fingerprint_lines(real)
        for c in trace.tool_calls:
            payload = "" if c.result is None else json.dumps(c.result, ensure_ascii=False, default=str)
            for line in fingerprint:
                if line in payload:
                    return _result(self.name, 0.0, self.threshold,
                                   f"content leaked into result of {c.name}: {line[:60]!r}")
        for line in fingerprint:
            if line in trace.output:
                return _result(
                    self.name, 0.0, self.threshold,
                    "output contains reference content with no read — "
                    f"parametric-memory contamination: {line[:60]!r}",
                )
        return _result(self.name, 1.0, self.threshold, "untouched")


def reference_untouched(path: str, *, source: str | Path, threshold: float = 1.0) -> ReferenceUntouchedExpectation:
    """then(reference_untouched("references/endpoints.md", source="skills/reddit"))."""
    return ReferenceUntouchedExpectation(path=path, source=Path(source), threshold=threshold)
