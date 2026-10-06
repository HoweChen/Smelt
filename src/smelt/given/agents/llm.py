"""LLM client abstraction: the seam between SmeltAgent and concrete model providers.

- ``LLMClient`` protocol: one chat completion, optionally with tool specs;
- ``ScriptedLLM``: deterministic client replaying canned responses in order —
  the "fixed output" path for tests;
- ``OpenAIChatClient``: OpenAI-compatible API (optional dependency openai).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from smelt.tools import Tool


@dataclass(frozen=True)
class ToolCall:
    """A tool call the LLM decided to make."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LLMResponse:
    """Result of one completion: text content plus zero or more tool calls."""

    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()

    @classmethod
    def say(cls, content: str) -> LLMResponse:
        return cls(content=content)

    @classmethod
    def call(cls, name: str, arguments: Mapping[str, Any] | None = None) -> LLMResponse:
        return cls(tool_calls=(ToolCall(name=name, arguments=dict(arguments or {})),))


@runtime_checkable
class LLMClient(Protocol):
    """Synchronous chat completion client.

    messages is an OpenAI-style list of dicts; tools are the tools available
    for this turn. Returns LLMResponse — non-empty content is treated as the
    user-facing answer; non-empty tool_calls are handed to the agent for
    execution before the conversation continues.
    """

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: Sequence[Tool],
    ) -> LLMResponse: ...


class ScriptedLLM:
    """Scripted LLM: emits preset responses in order, repeating the last one.

    Use it to exercise SmeltAgent's tool loop end to end without network access —
    also the deterministic path for "keep fixed output".
    """

    def __init__(self, responses: Sequence[LLMResponse | str]) -> None:
        if not responses:
            raise ValueError("ScriptedLLM requires at least one response")
        self._responses = [r if isinstance(r, LLMResponse) else LLMResponse.say(r) for r in responses]
        self._index = 0
        self.calls: list[list[dict[str, Any]]] = []  # messages seen per call, for debugging

    def complete(self, messages: list[dict[str, Any]], tools: Sequence[Tool]) -> LLMResponse:
        self.calls.append(list(messages))
        response = self._responses[min(self._index, len(self._responses) - 1)]
        self._index += 1
        return response


class OpenAIChatClient:
    """OpenAI-compatible chat.completions client (optional dependency openai).

    Usage::

        llm = OpenAIChatClient(model="kimi-k2", base_url="https://api.moonshot.cn/v1")
    """

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        **client_kwargs: Any,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError("OpenAIChatClient requires the optional dependency: uv add 'smelt[openai]'") from e
        kwargs: dict[str, Any] = {"model": model, "temperature": temperature}
        if base_url is not None:
            client_kwargs["base_url"] = base_url
        if api_key is not None:
            client_kwargs["api_key"] = api_key
        self._client = OpenAI(**client_kwargs)
        self._default = kwargs

    def complete(self, messages: list[dict[str, Any]], tools: Sequence[Tool]) -> LLMResponse:
        request: dict[str, Any] = {**self._default, "messages": messages}
        if tools:
            request["tools"] = [
                {"type": "function", "function": t.spec()} for t in tools
            ]
            request["tool_choice"] = "auto"
        completion = self._client.chat.completions.create(**request)
        message = completion.choices[0].message
        calls = tuple(
            ToolCall(
                name=c.function.name,
                arguments=json.loads(c.function.arguments or "{}"),
            )
            for c in (message.tool_calls or [])
        )
        return LLMResponse(content=message.content or "", tool_calls=calls)
