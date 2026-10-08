"""tests/test_llm_config.py"""
import pytest

from smelt.llm_config import LLMConfig, LLMConfigError


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in list(__import__("os").environ):
        if key.startswith("SMELT_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr("smelt.llm_config._auto_load", lambda: None)


def test_role_env_wins_over_legacy(monkeypatch):
    monkeypatch.setenv("SMELT_DOCTOR_MODEL", "claude-sonnet-4-5")
    monkeypatch.setenv("SMELT_DOCTOR_PROVIDER", "anthropic")
    monkeypatch.setenv("SMELT_API_KEY", "shared-key")
    cfg = LLMConfig.from_role("doctor")
    assert cfg.model == "claude-sonnet-4-5"
    assert cfg.provider == "anthropic"
    assert cfg.api_key == "shared-key"  # legacy shared fallback


def test_explicit_args_win(monkeypatch):
    monkeypatch.setenv("SMELT_DOCTOR_MODEL", "env-model")
    cfg = LLMConfig.from_role("doctor", model="explicit-model")
    assert cfg.model == "explicit-model"


def test_legacy_shared_keys_are_common_fallback(monkeypatch):
    monkeypatch.setenv("SMELT_BASE_URL", "https://proxy.example/v1")
    monkeypatch.setenv("SMELT_LLM_PROVIDER", "openai")
    cfg = LLMConfig.from_role("judge", model="kimi-k2")
    assert cfg.base_url == "https://proxy.example/v1"
    assert cfg.provider == "openai"


def test_unknown_role_rejected():
    with pytest.raises(ValueError, match="unknown role"):
        LLMConfig.from_role("wizard")


def test_build_requires_model():
    with pytest.raises(LLMConfigError, match="model"):
        LLMConfig.from_role("doctor").build()


def test_build_openai_client(monkeypatch):
    pytest.importorskip("openai")
    monkeypatch.setenv("SMELT_DOCTOR_API_KEY", "sk-test")
    client = LLMConfig.from_role("doctor", model="kimi-k2").build()
    from smelt.given.agents.llm import OpenAIChatClient
    assert isinstance(client, OpenAIChatClient)
