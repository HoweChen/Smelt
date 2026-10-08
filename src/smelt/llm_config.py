"""Role-based LLM configuration: every role gets its own
provider / base_url / api_key / model quartet.

Env resolution per key (explicit argument always wins)::

    SMELT_<ROLE>_<KEY>  >  SMELT_<KEY> (legacy shared)  >  provider default

Legacy shared keys: SMELT_LLM_PROVIDER, SMELT_BASE_URL, SMELT_API_KEY.
``model`` has no legacy fallback — except JUDGE, whose historical
SMELT_JUDGE_MODEL is exactly the role-style key.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from smelt.env import _auto_load
from smelt.given.agents.llm import LLMClient

ROLES = ("judge", "agent", "doctor", "challenger")

_LEGACY = {"provider": "SMELT_LLM_PROVIDER", "base_url": "SMELT_BASE_URL", "api_key": "SMELT_API_KEY"}


class LLMConfigError(ValueError):
    """A role's LLM configuration is unusable (model unresolvable)."""


@dataclass(frozen=True)
class LLMConfig:
    provider: str = "openai"  # "openai" | "anthropic"
    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None

    @classmethod
    def from_role(
        cls,
        role: str,
        *,
        provider: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ) -> LLMConfig:
        _auto_load()
        r = role.lower()
        if r not in ROLES:
            raise ValueError(f"unknown role {role!r}: expected one of {ROLES}")
        prefix = f"SMELT_{r.upper()}_"

        def pick(key: str, explicit: str | None, default: str | None = None) -> str | None:
            if explicit is not None:
                return explicit
            return os.environ.get(prefix + key) or os.environ.get(_LEGACY.get(key.lower(), "")) or default

        return cls(
            provider=pick("PROVIDER", provider, "openai"),
            base_url=pick("BASE_URL", base_url),
            api_key=pick("API_KEY", api_key),
            model=model or os.environ.get(prefix + "MODEL"),
        )

    def build(self) -> LLMClient:
        if not self.model:
            raise LLMConfigError(
                "model not configured: set SMELT_<ROLE>_MODEL in .env or pass an explicit model"
            )
        if self.provider == "openai":
            from smelt.given.agents.llm import OpenAIChatClient

            return OpenAIChatClient(self.model, base_url=self.base_url, api_key=self.api_key)
        if self.provider == "anthropic":
            from smelt.given.agents.langchain_llm import LangChainLLM

            return LangChainLLM.from_provider(
                "anthropic", self.model, base_url=self.base_url, api_key=self.api_key
            )
        raise LLMConfigError(f"unknown provider {self.provider!r}: expected 'openai' or 'anthropic'")
