"""then expectations: tool calls, JSON schema, strings, similarity to a reference answer.

Each expectation yields a 0~1 score, judged against its own pass threshold.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Protocol, runtime_checkable

from smelt.results import ExpectationResult
from smelt.trace import Trace


@runtime_checkable
class Expectation(Protocol):
    """then-expectation protocol. Implement it to define custom assertions."""

    name: str
    threshold: float

    def evaluate(self, trace: Trace) -> ExpectationResult: ...


def _result(name: str, score: float, threshold: float, message: str = "", **details: Any) -> ExpectationResult:
    return ExpectationResult(
        name=name,
        score=max(0.0, min(1.0, score)),
        threshold=threshold,
        message=message,
        details=details,
    )


# ---------------------------------------------------------------------------
# Commands / tool calls
# ---------------------------------------------------------------------------


def _args_match(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    """expected is a (recursive) subset of actual."""
    for key, value in expected.items():
        if key not in actual:
            return False
        got = actual[key]
        if isinstance(value, Mapping) and isinstance(got, Mapping):
            if not _args_match(got, value):
                return False
        elif got != value:
            return False
    return True


@dataclass(frozen=True)
class ToolCallExpectation:
    """Assert a tool was called (optionally with an argument subset match).

    Scoring: exact match 1.0; name matched but arguments mismatched 0.5 —
    distinguishing "right direction, wrong details" from "never went there".
    """

    tool_name: str
    args: Mapping[str, Any] | None = None
    threshold: float = 1.0

    @property
    def name(self) -> str:
        if self.args:
            return f"tool_call({self.tool_name}, args={dict(self.args)})"
        return f"tool_call({self.tool_name})"

    def evaluate(self, trace: Trace) -> ExpectationResult:
        calls = trace.calls_named(self.tool_name)
        if not calls:
            return _result(
                self.name, 0.0, self.threshold,
                f"{self.tool_name} was never called; actual calls: {trace.called_tools or '(none)'}",
            )
        if self.args is None:
            return _result(self.name, 1.0, self.threshold, f"{self.tool_name} called {len(calls)} time(s)")
        for call in calls:
            if _args_match(call.arguments, self.args):
                return _result(self.name, 1.0, self.threshold, "arguments matched", matched=dict(call.arguments))
        return _result(
            self.name, 0.5, self.threshold,
            f"{self.tool_name} was called but no arguments matched; expected subset: {dict(self.args)}",
            actual=[dict(c.arguments) for c in calls],
        )


@dataclass(frozen=True)
class NoToolCallExpectation:
    """Assert a tool was never called (optionally: never with an argument subset).

    ``args=None`` keeps name-only semantics; with ``args`` a violation is a call
    matching the name AND the argument subset."""

    tool_name: str
    args: Mapping[str, Any] | None = None
    threshold: float = 1.0

    @property
    def name(self) -> str:
        if self.args:
            return f"no_tool_call({self.tool_name}, args={dict(self.args)})"
        return f"no_tool_call({self.tool_name})"

    def evaluate(self, trace: Trace) -> ExpectationResult:
        calls = trace.calls_named(self.tool_name)
        if self.args is not None:
            calls = [c for c in calls if _args_match(c.arguments, self.args)]
        if calls:
            msg = f"{self.tool_name} was called {len(calls)} time(s)"
            if self.args:
                msg += f" with {dict(self.args)}"
            return _result(self.name, 0.0, self.threshold, msg)
        return _result(self.name, 1.0, self.threshold)


def tool_call(
    name: str,
    *,
    args: Mapping[str, Any] | None = None,
    threshold: float = 1.0,
) -> ToolCallExpectation:
    """then(tool_call("run_command", args={"cmd": "git status"}))."""
    return ToolCallExpectation(tool_name=name, args=args, threshold=threshold)


def no_tool_call(
    name: str,
    args: Mapping[str, Any] | None = None,
    *,
    threshold: float = 1.0,
) -> NoToolCallExpectation:
    """then(no_tool_call("read_file", args={"path": "references/x.md"}))."""
    return NoToolCallExpectation(tool_name=name, args=args, threshold=threshold)


# ---------------------------------------------------------------------------
# Budgets (operating envelopes): turns / tool calls / wall time
#
# Deterministic gates on the trace — no LLM involved. Quality can pass while
# the run is unaffordable ("budget burner"); these catch that. Binary scoring:
# within budget 1.0, exceeded 0.0.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TurnsUsedExpectation:
    """Assert the agent finished within a turn budget (agent loop iterations)."""

    max_turns: int
    threshold: float = 1.0

    @property
    def name(self) -> str:
        return f"turns_used(max={self.max_turns})"

    def evaluate(self, trace: Trace) -> ExpectationResult:
        if trace.turns <= self.max_turns:
            return _result(self.name, 1.0, self.threshold, f"{trace.turns} turn(s) within budget {self.max_turns}")
        return _result(
            self.name, 0.0, self.threshold,
            f"used {trace.turns} turns, over budget {self.max_turns}",
        )


@dataclass(frozen=True)
class ToolBudgetExpectation:
    """Assert a tool (or all tools together) was called at most ``max_calls`` times."""

    tool_name: str | None = None
    max_calls: int = 0
    threshold: float = 1.0

    @property
    def name(self) -> str:
        target = self.tool_name or "*"
        return f"tool_budget({target}, max={self.max_calls})"

    def evaluate(self, trace: Trace) -> ExpectationResult:
        calls = trace.calls_named(self.tool_name) if self.tool_name else trace.tool_calls
        count = len(calls)
        if count <= self.max_calls:
            return _result(self.name, 1.0, self.threshold, f"{count} call(s) within budget {self.max_calls}")
        return _result(
            self.name, 0.0, self.threshold,
            f"{count} call(s), over budget {self.max_calls}",
        )


@dataclass(frozen=True)
class WallTimeExpectation:
    """Assert the agent run finished within a wall-clock budget (seconds)."""

    max_seconds: float
    threshold: float = 1.0

    @property
    def name(self) -> str:
        return f"wall_time(max_seconds={self.max_seconds})"

    def evaluate(self, trace: Trace) -> ExpectationResult:
        if trace.wall_time_s is None:
            return _result(self.name, 1.0, self.threshold, "no timing recorded on the trace")
        if trace.wall_time_s <= self.max_seconds:
            return _result(
                self.name, 1.0, self.threshold,
                f"{trace.wall_time_s:.2f}s within budget {self.max_seconds}s",
            )
        return _result(
            self.name, 0.0, self.threshold,
            f"took {trace.wall_time_s:.2f}s, over budget {self.max_seconds}s",
        )


def turns_used(*, max: int, threshold: float = 1.0) -> TurnsUsedExpectation:
    """then(turns_used(max=5)) — fail when the agent needs more than 5 turns."""
    if max < 0:
        raise ValueError(f"max must be >= 0, got {max}")
    return TurnsUsedExpectation(max_turns=max, threshold=threshold)


def tool_budget(name: str | None = None, *, max: int, threshold: float = 1.0) -> ToolBudgetExpectation:
    """then(tool_budget("run_command", max=2)) — per-tool or total (no name) call cap."""
    if max < 0:
        raise ValueError(f"max must be >= 0, got {max}")
    return ToolBudgetExpectation(tool_name=name, max_calls=max, threshold=threshold)


def wall_time(*, max_seconds: float, threshold: float = 1.0) -> WallTimeExpectation:
    """then(wall_time(max_seconds=10)) — wall-clock budget for the agent run."""
    if max_seconds < 0:
        raise ValueError(f"max_seconds must be >= 0, got {max_seconds}")
    return WallTimeExpectation(max_seconds=max_seconds, threshold=threshold)


# ---------------------------------------------------------------------------
# Strings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OutputEqualsExpectation:
    expected: str
    strip: bool = True
    threshold: float = 1.0

    @property
    def name(self) -> str:
        return f"output_equals({self.expected[:40]!r})"

    def evaluate(self, trace: Trace) -> ExpectationResult:
        got = trace.output.strip() if self.strip else trace.output
        want = self.expected.strip() if self.strip else self.expected
        if got == want:
            return _result(self.name, 1.0, self.threshold)
        return _result(self.name, 0.0, self.threshold, f"actual output: {got[:200]!r}")


@dataclass(frozen=True)
class OutputContainsExpectation:
    substrings: tuple[str, ...]
    threshold: float = 1.0

    @property
    def name(self) -> str:
        return f"output_contains({list(self.substrings)!r})"

    def evaluate(self, trace: Trace) -> ExpectationResult:
        missing = [s for s in self.substrings if s not in trace.output]
        score = 1.0 - len(missing) / len(self.substrings) if self.substrings else 1.0
        message = "" if not missing else f"missing fragments: {missing}"
        return _result(self.name, score, self.threshold, message)


def output_equals(expected: str, *, strip: bool = True, threshold: float = 1.0) -> OutputEqualsExpectation:
    """then(output_equals("done")) — exact match on the final output."""
    return OutputEqualsExpectation(expected=expected, strip=strip, threshold=threshold)


def output_contains(*substrings: str, threshold: float = 1.0) -> OutputContainsExpectation:
    """then(output_contains("committed", "commit")) — scored by fragment ratio."""
    return OutputContainsExpectation(substrings=tuple(substrings), threshold=threshold)


# ---------------------------------------------------------------------------
# Similarity to a reference answer
# ---------------------------------------------------------------------------


def _default_similarity(a: str, b: str) -> float:
    """Built-in similarity: char-level SequenceMatcher, zero dependencies.
    Replaceable with an embedding-model scorer."""
    return SequenceMatcher(None, a, b).ratio()


@dataclass(frozen=True)
class TextSimilarExpectation:
    """Compare against a reference answer; passes when score >= threshold.

    ``scorer`` accepts any (actual, reference) -> 0~1 function
    (e.g. embedding-based semantic similarity); difflib is the default.
    """

    reference: str
    threshold: float = 0.8
    scorer: Callable[[str, str], float] = _default_similarity

    @property
    def name(self) -> str:
        return f"text_similar(reference={self.reference[:30]!r}, threshold={self.threshold})"

    def evaluate(self, trace: Trace) -> ExpectationResult:
        score = float(self.scorer(trace.output, self.reference))
        message = f"similarity {score:.2f}; actual output: {trace.output[:200]!r}"
        return _result(self.name, score, self.threshold, message, similarity=score)


def text_similar(
    reference: str,
    *,
    threshold: float | None = None,
    scorer: Callable[[str, str], float] | None = None,
) -> TextSimilarExpectation:
    """then(text_similar("reference answer...")).

    When threshold is omitted, the global default
    ``smelt.config.text_similar_threshold`` applies (initially 0.8; change via
    ``smelt.configure(text_similar_threshold=...)``).
    """
    from smelt.config import config

    kwargs: dict[str, Any] = {"scorer": scorer} if scorer is not None else {}
    return TextSimilarExpectation(
        reference=reference,
        threshold=threshold if threshold is not None else config.text_similar_threshold,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------


def _extract_json(output: str) -> Any:
    """Extract JSON from output: full parse → ```json fenced block → position-by-position
    scan for the first parseable {...} / [...]."""
    text_out = output.strip()
    try:
        return json.loads(text_out)
    except json.JSONDecodeError:
        pass
    fence = re.search(r"```(?:json)?\s*\n(.*?)```", text_out, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text_out):
        if ch in "{[":
            try:
                data, _ = decoder.raw_decode(text_out[i:])
                return data
            except json.JSONDecodeError:
                continue
    raise ValueError("no parseable JSON found in the output")


def _validate_minimal(data: Any, schema: Mapping[str, Any], path: str = "$") -> list[str]:
    """Dependency-free JSON Schema subset validation: type / required / properties / items / enum."""
    errors: list[str] = []
    type_map = {
        "object": dict, "array": list, "string": str,
        "integer": int, "number": (int, float), "boolean": bool, "null": type(None),
    }
    expected_type = schema.get("type")
    # bool is a subclass of int; exclude it from the integer check
    bool_masquerade = expected_type == "integer" and isinstance(data, bool)
    if expected_type in type_map and (bool_masquerade or not isinstance(data, type_map[expected_type])):
        errors.append(f"{path}: expected type {expected_type}, got {type(data).__name__}")
        return errors
    if "enum" in schema and data not in schema["enum"]:
        errors.append(f"{path}: {data!r} is not in enum {schema['enum']}")
    if isinstance(data, dict):
        for key in schema.get("required", []):
            if key not in data:
                errors.append(f"{path}: missing required field {key!r}")
        props = schema.get("properties", {})
        for key, sub in props.items():
            if key in data and isinstance(sub, Mapping):
                errors.extend(_validate_minimal(data[key], sub, f"{path}.{key}"))
    if isinstance(data, list) and isinstance(schema.get("items"), Mapping):
        for i, item in enumerate(data):
            errors.extend(_validate_minimal(item, schema["items"], f"{path}[{i}]"))
    return errors


def _validate_schema(data: Any, schema: Mapping[str, Any]) -> list[str]:
    """Prefer the jsonschema library (full spec); fall back to the built-in subset."""
    try:
        import jsonschema  # type: ignore
    except ImportError:
        return _validate_minimal(data, schema)
    validator = jsonschema.validators.validator_for(schema)(schema)
    return [f"{'$' + ''.join(f'[{p}]' if isinstance(p, int) else '.' + str(p) for p in e.absolute_path)}: {e.message}"
            for e in validator.iter_errors(data)]


@dataclass(frozen=True)
class JsonOutputExpectation:
    """Assert the final output is parseable JSON, optionally schema-validated /
    field-subset matched.

    Scoring: parseable 0.4; schema passed 0.4; contains subset matched 0.2.
    Weights normalize when only some parts are given.
    """

    schema: Mapping[str, Any] | None = None
    contains: Mapping[str, Any] | None = None
    threshold: float = 1.0

    @property
    def name(self) -> str:
        parts = []
        if self.schema is not None:
            parts.append("schema=…")
        if self.contains is not None:
            parts.append(f"contains={dict(self.contains)}")
        return f"json_output({', '.join(parts) or 'any json'})"

    def evaluate(self, trace: Trace) -> ExpectationResult:
        try:
            data = _extract_json(trace.output)
        except ValueError as e:
            return _result(self.name, 0.0, self.threshold, str(e))

        checks: list[tuple[float, list[str]]] = [(0.4, [])]  # base credit for parseability
        if self.schema is not None:
            checks.append((0.4, _validate_schema(data, self.schema)))
        if self.contains is not None:
            ok = isinstance(data, Mapping) and _args_match(data, self.contains)
            checks.append((0.2, [] if ok else [f"field subset mismatch, expected {dict(self.contains)}"]))

        total = sum(w for w, _ in checks)
        earned = sum(w for w, errs in checks if not errs)
        errors = [e for _, errs in checks for e in errs]
        score = earned / total
        return _result(self.name, score, self.threshold, "; ".join(errors), parsed=data)


def json_output(
    schema: Mapping[str, Any] | None = None,
    *,
    contains: Mapping[str, Any] | None = None,
    threshold: float = 1.0,
) -> JsonOutputExpectation:
    """then(json_output({"type": "object", "required": ["name"]})).

    - ``schema``: JSON Schema; full validation with the jsonschema package installed,
      built-in subset otherwise;
    - ``contains``: assert the output JSON contains the given field subset;
    - when both are given, schema validation weighs more.
    """
    return JsonOutputExpectation(schema=schema, contains=contains, threshold=threshold)
