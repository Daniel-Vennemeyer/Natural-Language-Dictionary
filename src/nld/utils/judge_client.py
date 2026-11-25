"""OpenAI-compatible client for the API rubric judge."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class JudgeClient:
    client: Any
    model_name: str
    provider: str


def get_judge_config(config: dict[str, Any] | None) -> dict[str, Any]:
    """The nested `judge` block of an evaluation config ({} if absent)."""
    nested = (config or {}).get("judge")
    return dict(nested) if isinstance(nested, dict) else {}


def build_judge_client(
    config: dict[str, Any] | None = None,
    *,
    default_model: str = "gpt-5.4",
) -> JudgeClient:
    """Client for judge.provider='openai' (the default). The key is read from the variable
    named by judge.api_key_env (default OPENAI_API_KEY); judge.base_url
    selects another OpenAI-compatible endpoint and judge.model overrides default_model."""
    judge_config = get_judge_config(config)
    provider = str(judge_config.get("provider", "openai")).strip().lower()
    if provider != "openai":
        raise ValueError(f"Unsupported judge provider {provider!r}. Expected 'openai'.")

    from openai import OpenAI

    api_key_env = str(judge_config.get("api_key_env", "OPENAI_API_KEY"))
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise EnvironmentError(f"{api_key_env} is required for judge provider 'openai'.")
    client_kwargs: dict[str, Any] = {"api_key": api_key}
    if judge_config.get("base_url"):
        client_kwargs["base_url"] = judge_config["base_url"]
    return JudgeClient(
        client=OpenAI(**client_kwargs),
        model_name=str(judge_config.get("model", judge_config.get("deployment", default_model))),
        provider=provider,
    )
