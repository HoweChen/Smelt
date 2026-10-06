"""Tests for .env support: parsing, loading, configurable location, SMELT_* fallbacks.

Conventions:
- env files hold ``SMELT_``-prefixed variables (SMELT_API_KEY / SMELT_BASE_URL /
  SMELT_JUDGE_MODEL);
- the file location is specified in code: ``smelt.configure(env_file="...")``,
  default ``.env`` in the working directory;
- variables already present in os.environ always win (never overridden).
"""

import os
import sys
import types
from dataclasses import replace

import pytest

import smelt
import smelt.config
import smelt.env as env_module
from smelt.env import load_env, parse_env

# smelt/__init__ rebinds the name "config" to the SmeltConfig instance, so the
# module must be fetched from sys.modules.
config_module = sys.modules["smelt.config"]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOOD = os.path.join(ROOT, "examples", "good_skill")


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch, tmp_path):
    """Every test starts with a clean slate: no SMELT_* vars, no auto-load,
    env_file pointing at a nonexistent path (deterministic, never touches a
    real .env). Environment variables are saved/restored manually because
    load_env writes os.environ directly (outside monkeypatch's tracking)."""
    keys = ("SMELT_API_KEY", "SMELT_BASE_URL", "SMELT_JUDGE_MODEL", "SMELT_TEST_VAR")
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


def _env_file(tmp_path, content: str):
    path = tmp_path / "test.env"
    path.write_text(content, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_parse_env_basic_forms():
    parsed = parse_env(
        """# a comment line
SMELT_API_KEY=sk-123
export SMELT_BASE_URL="https://api.example.com/v1"
SINGLE='quoted value'
EMPTY=
SPACED = spaced value
INLINE=val # trailing comment
"""
    )
    assert parsed == {
        "SMELT_API_KEY": "sk-123",
        "SMELT_BASE_URL": "https://api.example.com/v1",
        "SINGLE": "quoted value",
        "EMPTY": "",
        "SPACED": "spaced value",
        "INLINE": "val",
    }


def test_parse_env_ignores_garbage_lines():
    assert parse_env("not a kv line\n\n   \nOK=1\n") == {"OK": "1"}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def test_load_env_applies_variables(tmp_path):
    path = _env_file(tmp_path, "SMELT_TEST_VAR=from-file\n")
    applied = load_env(path)
    assert applied == {"SMELT_TEST_VAR": "from-file"}
    assert os.environ["SMELT_TEST_VAR"] == "from-file"


def test_load_env_never_overrides_existing_environ(tmp_path, monkeypatch):
    monkeypatch.setenv("SMELT_TEST_VAR", "from-shell")
    path = _env_file(tmp_path, "SMELT_TEST_VAR=from-file\nSMELT_API_KEY=sk-new\n")
    applied = load_env(path)
    assert applied == {"SMELT_API_KEY": "sk-new"}  # only the new one applied
    assert os.environ["SMELT_TEST_VAR"] == "from-shell"


def test_load_env_missing_file_returns_empty(tmp_path):
    assert load_env(tmp_path / "ghost.env") == {}


def test_configured_env_file_location_used_by_default(tmp_path):
    path = _env_file(tmp_path, "SMELT_TEST_VAR=from-configured-path\n")
    smelt.configure(env_file=str(path))
    applied = smelt.load_env()
    assert applied == {"SMELT_TEST_VAR": "from-configured-path"}
    assert config_module.config.env_file == str(path)


def test_env_file_none_disables_loading():
    smelt.configure(env_file=None)
    assert smelt.load_env() == {}


# ---------------------------------------------------------------------------
# OpenAIChatClient fallbacks (SMELT_ prefix, never MOONSHOT_)
# ---------------------------------------------------------------------------


def _fake_openai(monkeypatch, captured):
    fake_message = types.SimpleNamespace(content="done", tool_calls=None)

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=fake_message)])

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured["client_kwargs"] = kwargs
            self.chat = types.SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    from smelt.given.agents.llm import OpenAIChatClient

    return OpenAIChatClient


def test_openai_client_reads_smelt_env_fallbacks(monkeypatch):
    monkeypatch.setenv("SMELT_API_KEY", "sk-from-env")
    monkeypatch.setenv("SMELT_BASE_URL", "https://env.example.com/v1")
    captured = {}
    client_cls = _fake_openai(monkeypatch, captured)
    client_cls("kimi-k2")
    assert captured["client_kwargs"] == {"api_key": "sk-from-env", "base_url": "https://env.example.com/v1"}


def test_openai_client_explicit_args_win_over_env(monkeypatch):
    monkeypatch.setenv("SMELT_API_KEY", "sk-from-env")
    captured = {}
    client_cls = _fake_openai(monkeypatch, captured)
    client_cls("kimi-k2", api_key="sk-explicit", base_url="https://explicit.example.com/v1")
    assert captured["client_kwargs"] == {"api_key": "sk-explicit", "base_url": "https://explicit.example.com/v1"}


def test_openai_client_auto_loads_configured_env_file(monkeypatch, tmp_path):
    path = _env_file(tmp_path, "SMELT_API_KEY=sk-from-dotenv\n")
    smelt.configure(env_file=str(path))
    captured = {}
    client_cls = _fake_openai(monkeypatch, captured)
    client_cls("kimi-k2")
    assert captured["client_kwargs"].get("api_key") == "sk-from-dotenv"


def test_openai_client_omits_unset_fallbacks(monkeypatch):
    captured = {}
    client_cls = _fake_openai(monkeypatch, captured)
    client_cls("kimi-k2")
    assert captured["client_kwargs"] == {}


# ---------------------------------------------------------------------------
# CLI: judge model from SMELT_JUDGE_MODEL
# ---------------------------------------------------------------------------


def test_cli_evaluate_reads_judge_model_from_env(monkeypatch, capsys):
    monkeypatch.setenv("SMELT_JUDGE_MODEL", "judge-from-env")
    writing_json = '{"dimensions": [{"name": "d", "score": 0.9, "comment": "c"}], "overall_comment": "ok"}'
    captured = {}
    fake_message = types.SimpleNamespace(content=writing_json, tool_calls=None)

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=fake_message)])

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = types.SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))

    from smelt.cli import main

    code = main(["evaluate", GOOD, "--no-lint", "--no-suggestions"])
    assert code == 0
    assert captured["model"] == "judge-from-env"
    assert "no --judge-model" not in capsys.readouterr().err
