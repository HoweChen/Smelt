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


config = SmeltConfig()


def configure(
    *,
    text_similar_threshold: float | None = None,
    llm_judge_threshold: float | None = None,
) -> SmeltConfig:
    """Update global default thresholds and return the updated config.

    Only values in [0, 1] are accepted; unset (None) fields keep their values.
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
    config = replace(config, **updates)
    return config
