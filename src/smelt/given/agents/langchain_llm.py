"""langchain bridge: wrap any langchain chat model as a smelt LLMClient.

Positioning: smelt's core stays dependency-free; langchain is only an optional
adapter (``smelt[langchain]``) so testers with existing langchain model configs
can reuse them directly. langgraph is intentionally not used — judge and agent
both need just "one completion", not a graph orchestration runtime.

Usage::

    from langchain_openai import ChatOpenAI
    from smelt.given.agents.langchain_llm import LangChainLLM

    llm = LangChainLLM(ChatOpenAI(model="kimi-k2", base_url="..."))
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Sequence
from typing import Any

from smelt.given.agents.llm import LLMResponse, ToolCall
from smelt.tools import Tool

try:
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
except ImportError as e:  # pragma: no cover - optional dependency
    raise ImportError(
        "LangChainLLM requires the optional dependency: uv add 'smelt[langchain]'"
    ) from e


def _parse_tool_call(t: dict[str, Any]) -> tuple[str, dict[str, Any], str | None]:
    """Accept the OpenAI wire shape ({id, type, function:{name, arguments}}) and
    the legacy flat shape ({name, arguments}); return (name, args, id)."""
    if "function" in t:
        args = t["function"].get("arguments") or {}
        if isinstance(args, str):
            args = json.loads(args)
        return t["function"]["name"], args, t.get("id")
    return t["name"], t.get("arguments", {}), t.get("id")


def _to_langchain(messages: list[dict[str, Any]]) -> list[BaseMessage]:
    """OpenAI-style dicts → langchain messages. Existing tool-call ids are kept;
    missing ones are synthesized and paired with the following tool messages."""
    counter = itertools.count(1)
    pending_ids: list[str] = []
    out: list[BaseMessage] = []
    for m in messages:
        role = m["role"]
        if role == "system":
            out.append(SystemMessage(content=m["content"]))
        elif role == "user":
            out.append(HumanMessage(content=m["content"]))
        elif role == "assistant":
            calls = []
            for t in (m.get("tool_calls") or []):
                name, args, call_id = _parse_tool_call(t)
                calls.append({"name": name, "args": args, "id": call_id or f"smelt-call-{next(counter)}"})
            pending_ids.extend(c["id"] for c in calls)
            out.append(AIMessage(content=m.get("content") or "", tool_calls=calls))
        elif role == "tool":
            call_id = m.get("tool_call_id")
            if call_id and call_id in pending_ids:
                pending_ids.remove(call_id)
            else:
                call_id = pending_ids.pop(0) if pending_ids else "smelt-call-0"
            out.append(ToolMessage(
                content=m["content"],
                tool_call_id=call_id,
                name=m.get("name"),
            ))
        else:
            raise ValueError(f"unsupported message role: {role}")
    return out


class LangChainLLM:
    """Adapt a langchain BaseChatModel into an LLMClient.

    ``from_provider`` / ``from_env`` construct the chat model for you —
    provider mode, optional base_url (provider default when omitted), and
    api_key (explicit arg > SMELT_API_KEY from .env > langchain's own env
    default).
    """

    def __init__(self, chat_model: Any) -> None:
        self._model = chat_model

    @classmethod
    def from_provider(
        cls,
        provider: str,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> LangChainLLM:
        """Build a chat model for "openai" or "anthropic" in base_url mode.

        ``base_url`` / ``api_key`` fall back to SMELT_BASE_URL / SMELT_API_KEY
        (auto-loaded from the configured .env); when both are unset the
        provider's default endpoint and langchain's own env keys apply.
        """
        import os

        from smelt.env import _auto_load

        _auto_load()
        api_key = api_key or os.environ.get("SMELT_API_KEY")
        base_url = base_url or os.environ.get("SMELT_BASE_URL")

        params: dict[str, Any] = {"model": model, **kwargs}
        if base_url is not None:
            params["base_url"] = base_url
        if api_key is not None:
            params["api_key"] = api_key

        if provider == "openai":
            try:
                from langchain_openai import ChatOpenAI
            except ImportError as e:
                raise ImportError(
                    "provider='openai' requires langchain-openai: uv add 'smelt[langchain-openai]'"
                ) from e
            return cls(ChatOpenAI(**params))
        if provider == "anthropic":
            try:
                from langchain_anthropic import ChatAnthropic
            except ImportError as e:
                raise ImportError(
                    "provider='anthropic' requires langchain-anthropic: uv add 'smelt[langchain-anthropic]'"
                ) from e
            return cls(ChatAnthropic(**params))
        raise ValueError(f"unknown provider {provider!r}: expected 'openai' or 'anthropic'")

    @classmethod
    def from_env(cls, model: str, **kwargs: Any) -> LangChainLLM:
        """Build from the configured .env / shell env — three keys only::

            SMELT_LLM_PROVIDER=openai        # or anthropic (default: openai)
            SMELT_BASE_URL=https://...       # optional; omitted → provider default
            SMELT_API_KEY=sk-...
        """
        import os

        from smelt.env import _auto_load

        _auto_load()
        return cls.from_provider(
            os.environ.get("SMELT_LLM_PROVIDER", "openai"),
            model,
            base_url=os.environ.get("SMELT_BASE_URL"),
            api_key=os.environ.get("SMELT_API_KEY"),
            **kwargs,
        )

    def complete(self, messages: list[dict[str, Any]], tools: Sequence[Tool]) -> LLMResponse:
        model = self._model
        if tools:
            try:
                model = model.bind_tools([
                    {"type": "function", "function": t.spec()} for t in tools
                ])
            except NotImplementedError:
                # Some models (e.g. test fakes) do not support tool binding; call as-is
                pass
        ai_message = model.invoke(_to_langchain(messages))
        content = ai_message.content
        if isinstance(content, list):  # some models return content-block lists
            content = "".join(
                block.get("text", "") if isinstance(block, dict) else str(block) for block in content
            )
        calls = tuple(
            ToolCall(name=c["name"], arguments=dict(c.get("args", {})))
            for c in (getattr(ai_message, "tool_calls", None) or [])
        )
        return LLMResponse(content=content or "", tool_calls=calls)
