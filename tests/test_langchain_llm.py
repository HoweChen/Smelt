"""Tests for the LangChainLLM adapter (using langchain-core's fake models, no network)."""

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from smelt import LLMClient, new_case, output_equals, smelt_agent, text, tool, tool_call
from smelt.given.agents.langchain_llm import LangChainLLM, _to_langchain


def test_converts_all_message_roles():
    lc = _to_langchain([
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "", "tool_calls": [{"name": "t", "arguments": {"a": 1}}]},
        {"role": "tool", "name": "t", "content": "result"},
    ])
    assert [m.type for m in lc] == ["system", "human", "ai", "tool"]
    # the assistant's tool call and the following tool message share the synthesized id
    assert lc[2].tool_calls[0]["id"] == lc[3].tool_call_id
    assert lc[2].tool_calls[0]["args"] == {"a": 1}


def test_unsupported_role_raises():
    with pytest.raises(ValueError, match="unsupported message role"):
        _to_langchain([{"role": "function", "content": "x"}])


def test_plain_text_response():
    model = GenericFakeChatModel(messages=iter([AIMessage(content="hello")]))
    resp = LangChainLLM(model).complete([{"role": "user", "content": "hi"}], [])
    assert resp.content == "hello" and resp.tool_calls == ()


def test_tool_call_response_maps_to_llm_response():
    ai = AIMessage(
        content="",
        tool_calls=[{"name": "write_file", "args": {"path": "a.txt", "content": "1"}, "id": "call-1"}],
    )
    model = GenericFakeChatModel(messages=iter([ai]))

    @tool
    def write_file(path: str, content: str) -> str:
        """Write a file"""
        return "ok"

    resp = LangChainLLM(model).complete([{"role": "user", "content": "hi"}], [write_file])
    assert resp.tool_calls[0].name == "write_file"
    assert resp.tool_calls[0].arguments == {"path": "a.txt", "content": "1"}


def test_content_block_list_is_joined():
    ai = AIMessage(content=[{"type": "text", "text": "part one "}, "part two"])
    model = GenericFakeChatModel(messages=iter([ai]))
    resp = LangChainLLM(model).complete([{"role": "user", "content": "hi"}], [])
    assert resp.content == "part one part two"


def test_langchain_llm_drives_smelt_agent_end_to_end():
    ai_call = AIMessage(
        content="",
        tool_calls=[{"name": "echo", "args": {"text": "ping"}, "id": "c1"}],
    )
    ai_final = AIMessage(content="pong")
    model = GenericFakeChatModel(messages=iter([ai_call, ai_final]))

    @tool
    def echo(text: str) -> str:
        """Echo"""
        return text.upper()

    result = (
        new_case("lc-agent")
        .given(smelt_agent(llm=LangChainLLM(model), tools=[echo], system_prompt="sys"))
        .when(text("ping it"))
        .then(tool_call("echo", args={"text": "ping"}))
        .then(output_equals("pong"))
        .run()
    )
    assert result.passed, result.summary()
    # the tool result was fed back as a ToolMessage with the synthesized id paired
    assert result.trace.tool_calls[0].result == "PING"


def test_langchain_llm_satisfies_protocol():
    model = GenericFakeChatModel(messages=iter([AIMessage(content="x")]))
    assert isinstance(LangChainLLM(model), LLMClient)
