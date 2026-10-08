""".env support: smelt reads SMELT_* variables from a .env file.

- Location is specified in code: ``smelt.configure(env_file="path/to/.env")``
  (default ``.env`` in the working directory; ``None`` disables loading);
- ``load_env()`` parses and applies the file into ``os.environ`` — variables
  already present in the environment always win (never overridden);
- Role-based quartet (primary scheme): ``SMELT_<ROLE>_PROVIDER`` /
  ``_BASE_URL`` / ``_API_KEY`` / ``_MODEL`` for ROLE in
  JUDGE / AGENT / DOCTOR / CHALLENGER (see ``smelt.llm_config.LLMConfig``);
- Legacy shared keys ``SMELT_API_KEY`` / ``SMELT_BASE_URL`` /
  ``SMELT_LLM_PROVIDER`` remain as the common fallback;
  ``OpenAIChatClient`` auto-loads the configured file once.

Zero dependencies — the parser covers the common .env subset: comments,
``export`` prefix, single/double quotes, inline comments on unquoted values.
"""

from __future__ import annotations

import os
from pathlib import Path

_auto_loaded = False  # auto-load fires once per process (tests reset this flag)


def parse_env(text: str) -> dict[str, str]:
    """Parse .env content into a dict. Lines without KEY=VALUE are ignored."""
    result: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].rstrip()  # inline comment on unquoted values
        result[key] = value
    return result


def load_env(path: str | os.PathLike[str] | None = None, *, override: bool = False) -> dict[str, str]:
    """Load a .env file into os.environ; returns the variables actually applied.

    ``path`` defaults to ``smelt.config.env_file``; a missing file or a None
    location is not an error — it simply applies nothing. Existing environment
    variables win unless ``override=True``.
    """
    if path is None:
        from smelt.config import config  # lazy: reads the live singleton

        path = config.env_file
    if path is None:
        return {}
    file = Path(path)
    if not file.is_file():
        return {}
    applied: dict[str, str] = {}
    for key, value in parse_env(file.read_text(encoding="utf-8")).items():
        if override or key not in os.environ:
            os.environ[key] = value
            applied[key] = value
    return applied


def _auto_load() -> None:
    """Load the configured .env once per process (OpenAIChatClient / CLI call this)."""
    global _auto_loaded
    if not _auto_loaded:
        _auto_loaded = True
        load_env()
