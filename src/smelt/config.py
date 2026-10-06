"""Global configuration: project-wide defaults such as assertion thresholds.

Usage::

    import smelt

    smelt.configure(text_similar_threshold=0.7, llm_judge_threshold=0.75)

After that, ``text_similar("...")`` and ``llm_judge(...)`` use these defaults when
no explicit threshold is passed; an explicit threshold always wins.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass
class SmeltConfig:
    """Smelt's global configuration (in-process singleton, see ``config``)."""

    text_similar_threshold: float = 0.8
    llm_judge_threshold: float = 0.8
    env_file: str | None = ".env"  # .env location (SMELT_* variables); None disables loading


config = SmeltConfig()

_UNSET = object()  # distinguishes "leave unchanged" from an explicit None


def configure(
    *,
    text_similar_threshold: float | None = None,
    llm_judge_threshold: float | None = None,
    env_file: str | None | object = _UNSET,
) -> SmeltConfig:
    """Update global defaults and return the updated config.

    Only thresholds in [0, 1] are accepted; unset (None) fields keep their
    values. ``env_file`` accepts a path string, or None to disable .env loading.
    """
    global config
    updates = {}
    for key, value in (
        ("text_similar_threshold", text_similar_threshold),
        ("llm_judge_threshold", llm_judge_threshold),
    ):
        if value is not None:
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{key} must be within [0, 1], got {value}")
            updates[key] = value
    if env_file is not _UNSET:
        updates["env_file"] = env_file
    config = replace(config, **updates)
    return config
