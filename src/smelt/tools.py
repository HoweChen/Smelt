"""Tool definitions: turn plain Python functions into agent-callable, trace-assertable tools."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, get_type_hints

_PY_TO_JSON = {
    str: {"type": "string"},
    int: {"type": "integer"},
    float: {"type": "number"},
    bool: {"type": "boolean"},
    list: {"type": "array"},
    dict: {"type": "object"},
}


def _schema_for(fn: Callable[..., Any]) -> dict[str, Any]:
    """Derive a JSON Schema from the signature and type hints (common scalars only)."""
    hints = get_type_hints(fn)
    sig = inspect.signature(fn)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        annotation = hints.get(name)
        properties[name] = dict(_PY_TO_JSON.get(annotation, {}))
        if param.default is inspect.Parameter.empty:
            required.append(name)
    return {"type": "object", "properties": properties, "required": required}


@dataclass
class Tool:
    """A tool the agent can call."""

    name: str
    description: str
    handler: Callable[..., Any]
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.parameters:
            self.parameters = _schema_for(self.handler)

    def invoke(self, arguments: Mapping[str, Any]) -> Any:
        return self.handler(**dict(arguments))

    def spec(self) -> dict[str, Any]:
        """OpenAI-style tool description for the LLM."""
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


def tool(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Tool | Callable[[Callable[..., Any]], Tool]:
    """Turn a function into a Tool. Supports both @tool and @tool(name="...").

    The description defaults to the first docstring line; the parameter schema is
    derived from type annotations and can be overridden at construction time.
    """

    def wrap(f: Callable[..., Any]) -> Tool:
        doc = (inspect.getdoc(f) or "").splitlines()
        return Tool(
            name=name or f.__name__,
            description=description or (doc[0] if doc else f.__name__),
            handler=f,
        )

    return wrap(fn) if fn is not None else wrap


def mock_tool(base: Tool, results: Mapping[str, Any]) -> Tool:
    """Wrap a tool: when any string argument path-matches a key in ``results``,
    return the mapped value instead of calling the real handler. For
    content-level robustness tests (empty / broken / stale reference content) —
    NOT for with/without ablation (there, simply don't mount)."""
    from smelt.refpath import ref_path_matches

    def handler(**kwargs: Any) -> Any:
        for key, value in results.items():
            if any(isinstance(v, str) and ref_path_matches(v, key) for v in kwargs.values()):
                return value
        return base.handler(**kwargs)

    return Tool(name=base.name, description=base.description, handler=handler,
                parameters=dict(base.parameters))
