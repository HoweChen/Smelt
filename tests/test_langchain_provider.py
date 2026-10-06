"""Tests for LangChainLLM.from_provider / from_env: provider factory with
base_url mode for OpenAI and Anthropic.

Config-file contract (.env): three keys only —
  SMELT_LLM_PROVIDER  ("openai" | "anthropic")
  SMELT_BASE_URL      (optional; omitted → provider default)
  SMELT_API_KEY
"""

import os
import sys
import types
from dataclasses import replace

import pytest

import smelt
import smelt.config
import smelt.env as env_module
from smelt.given.agents.langchain_llm import LangChainLLM

config_module = sys.modules["smelt.config"]


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch, tmp_path):
    keys = ("SMELT_API_KEY", "SMELT_BASE_URL", "SMELT_LLM_PROVIDER")
    saved = {k: os.environ.get(k) for k in keys}
    for k in keys:
        os.environ.pop(k, None)
    monkeypatch.setattr(config_module, "config", replace(config_module.config, env_file=str(tmp_path / "none.env")))
    monkeypatch.setattr(env_module, "_auto_loaded", False)
    yield tmp_path
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _fake_provider_modules(monkeypatch, captured):
    """Register fake langchain_openai / langchain_anthropic recording init kwargs."""

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured["openai"] = kwargs

    class FakeChatAnthropic:
        def __init__(self, **kwargs):
            captured["anthropic"] = kwargs

    monkeypatch.setitem(sys.modules, "langchain_openai", types.SimpleNamespace(ChatOpenAI=FakeChatOpenAI))
    monkeypatch.setitem(sys.modules, "langchain_anthropic", types.SimpleNamespace(ChatAnthropic=FakeChatAnthropic))
    return captured


def test_from_provider_openai_with_base_url(monkeypatch):
    captured = _fake_provider_modules(monkeypatch, {})
    llm = LangChainLLM.from_provider("openai", "kimi-k2", base_url="https://api.moonshot.cn/v1", api_key="sk-x")
    assert isinstance(llm, LangChainLLM)
    assert captured["openai"] == {
        "model": "kimi-k2",
        "base_url": "https://api.moonshot.cn/v1",
        "api_key": "sk-x",
    }


def test_from_provider_omits_base_url_for_provider_default(monkeypatch):
    captured = _fake_provider_modules(monkeypatch, {})
    LangChainLLM.from_provider("openai", "gpt-5", api_key="sk-x")
    assert "base_url" not in captured["openai"]  # provider default endpoint

    LangChainLLM.from_provider("anthropic", "claude-sonnet-4-5", api_key="sk-y")
    assert "base_url" not in captured["anthropic"]


def test_from_provider_anthropic_with_base_url(monkeypatch):
    captured = _fake_provider_modules(monkeypatch, {})
    LangChainLLM.from_provider("anthropic", "claude-sonnet-4-5", base_url="https://proxy.example.com", api_key="sk-y")
    assert captured["anthropic"]["base_url"] == "https://proxy.example.com"


def test_from_provider_falls_back_to_smelt_env(monkeypatch):
    monkeypatch.setenv("SMELT_API_KEY", "sk-from-env")
    monkeypatch.setenv("SMELT_BASE_URL", "https://env.example.com/v1")
    captured = _fake_provider_modules(monkeypatch, {})
    LangChainLLM.from_provider("openai", "kimi-k2")
    assert captured["openai"]["api_key"] == "sk-from-env"
    assert captured["openai"]["base_url"] == "https://env.example.com/v1"


def test_from_provider_explicit_args_win_over_env(monkeypatch):
    monkeypatch.setenv("SMELT_API_KEY", "sk-from-env")
    captured = _fake_provider_modules(monkeypatch, {})
    LangChainLLM.from_provider("anthropic", "claude-sonnet-4-5", api_key="sk-explicit")
    assert captured["anthropic"]["api_key"] == "sk-explicit"


def test_from_provider_unknown_provider_raises(monkeypatch):
    _fake_provider_modules(monkeypatch, {})
    with pytest.raises(ValueError, match="unknown provider"):
        LangChainLLM.from_provider("bedrock", "m")


def test_from_provider_missing_package_message(monkeypatch):
    monkeypatch.setitem(sys.modules, "langchain_anthropic", None)  # import fails
    with pytest.raises(ImportError, match="langchain-anthropic"):
        LangChainLLM.from_provider("anthropic", "claude-sonnet-4-5", api_key="sk-y")


def test_from_env_reads_three_config_keys(monkeypatch):
    monkeypatch.setenv("SMELT_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("SMELT_BASE_URL", "https://proxy.example.com")
    monkeypatch.setenv("SMELT_API_KEY", "sk-env")
    captured = _fake_provider_modules(monkeypatch, {})
    llm = LangChainLLM.from_env("claude-sonnet-4-5")
    assert isinstance(llm, LangChainLLM)
    assert captured["anthropic"] == {
        "model": "claude-sonnet-4-5",
        "base_url": "https://proxy.example.com",
        "api_key": "sk-env",
    }


def test_from_env_defaults_provider_to_openai(monkeypatch):
    captured = _fake_provider_modules(monkeypatch, {})
    LangChainLLM.from_env("gpt-5")
    assert captured["openai"]["model"] == "gpt-5"
    assert "base_url" not in captured["openai"]
    assert "api_key" not in captured["openai"]  # langchain's own env default kicks in


def test_from_env_loads_dotenv_file(monkeypatch, tmp_path):
    env_file = tmp_path / "smelt.env"
    env_file.write_text(
        "SMELT_LLM_PROVIDER=anthropic\nSMELT_API_KEY=sk-from-dotenv\n", encoding="utf-8"
    )
    smelt.configure(env_file=str(env_file))
    captured = _fake_provider_modules(monkeypatch, {})
    LangChainLLM.from_env("claude-sonnet-4-5")
    assert captured["anthropic"]["api_key"] == "sk-from-dotenv"
    assert "base_url" not in captured["anthropic"]
