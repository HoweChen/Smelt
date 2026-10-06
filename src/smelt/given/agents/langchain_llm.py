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


def _to_langchain(messages: list[dict[str, Any]]) -> list[BaseMessage]:
    """OpenAI-style dicts → langchain messages. Tool-call ids are synthesized and paired."""
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
            calls = [
                {"name": t["name"], "args": t.get("arguments", {}), "id": f"smelt-call-{next(counter)}"}
                for t in (m.get("tool_calls") or [])
            ]
            pending_ids.extend(c["id"] for c in calls)
            out.append(AIMessage(content=m.get("content") or "", tool_calls=calls))
        elif role == "tool":
            out.append(ToolMessage(
                content=m["content"],
                tool_call_id=pending_ids.pop(0) if pending_ids else "smelt-call-0",
                name=m.get("name"),
            ))
        else:
            raise ValueError(f"unsupported message role: {role}")
    return out


class LangChainLLM:
    """Adapt a langchain BaseChatModel into an LLMClient."""

    def __init__(self, chat_model: Any) -> None:
        self._model = chat_model

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
