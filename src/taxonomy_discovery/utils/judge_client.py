from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from openai import OpenAI
except ImportError:                                              # local-judge paths need no openai pkg
    class OpenAI:  # noqa: N801 -- placeholder keeping module importable
        def __init__(self, *a, **k):
            raise ImportError("the 'openai' package is required for API judge providers: pip install openai")


AZURE_DEEPSEEK_V4_FLASH_PROVIDER = "azure_deepseek_v4_flash"
AZURE_DEEPSEEK_V4_FLASH_BASE_URL = (
    "https://YOUR-AZURE-RESOURCE.services.ai.azure.com/openai/v1/"
)
AZURE_DEEPSEEK_V4_FLASH_DEPLOYMENT = "DeepSeek-V4-Flash"
AZURE_DEEPSEEK_V4_FLASH_KEY_ENV = "AZURE_DEEPSEEK_V4_FLASH_API_KEY"


@dataclass(frozen=True)
class JudgeClient:
    client: Any
    model_name: str
    provider: str


def get_judge_config(config: dict[str, Any] | None) -> dict[str, Any]:
    """Resolve a nested shared judge block, with flat-key compatibility."""
    config = config or {}
    nested = config.get("judge")
    if isinstance(nested, dict):
        return dict(nested)

    return {
        key.removeprefix("judge_"): value
        for key, value in config.items()
        if key.startswith("judge_")
        and key
        not in {
            "judge_metrics",
            "judge_max_new_tokens",
            "judge_system_prompt",
        }
    }


def build_judge_client(
    config: dict[str, Any] | None = None,
    *,
    default_model: str = "gpt-5.4",
) -> JudgeClient:
    judge_config = get_judge_config(config)
    provider = str(judge_config.get("provider", "openai")).strip().lower()

    if provider == AZURE_DEEPSEEK_V4_FLASH_PROVIDER:
        base_url = str(
            judge_config.get("base_url", AZURE_DEEPSEEK_V4_FLASH_BASE_URL)
        )
        model_name = str(
            judge_config.get(
                "deployment",
                judge_config.get("model", AZURE_DEEPSEEK_V4_FLASH_DEPLOYMENT),
            )
        )
        api_key_env = str(
            judge_config.get("api_key_env", AZURE_DEEPSEEK_V4_FLASH_KEY_ENV)
        )
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise EnvironmentError(
                f"{api_key_env} is required for judge provider "
                f"{AZURE_DEEPSEEK_V4_FLASH_PROVIDER!r}."
            )
        return JudgeClient(
            client=OpenAI(base_url=base_url, api_key=api_key),
            model_name=model_name,
            provider=provider,
        )

    if provider == "openai":
        api_key_env = str(judge_config.get("api_key_env", "OPENAI_API_KEY"))
        api_key = os.getenv(api_key_env)
        if not api_key and api_key_env == "OPENAI_API_KEY":
            key_path = Path("key.txt")
            if key_path.exists():
                api_key = key_path.read_text(encoding="utf-8").strip()
        if not api_key:
            raise EnvironmentError(
                f"{api_key_env} is required for judge provider 'openai'."
            )
        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if judge_config.get("base_url"):
            client_kwargs["base_url"] = judge_config["base_url"]
        return JudgeClient(
            client=OpenAI(**client_kwargs),
            model_name=str(
                judge_config.get(
                    "model",
                    judge_config.get("deployment", default_model),
                )
            ),
            provider=provider,
        )

    raise ValueError(
        f"Unsupported judge provider {provider!r}. Expected 'openai' or "
        f"{AZURE_DEEPSEEK_V4_FLASH_PROVIDER!r}."
    )


def chat_completion(
    judge: JudgeClient,
    messages: list[dict[str, Any]],
    *,
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> Any:
    """Provider-robust chat completion.

    Different deployments accept different token/temperature parameters: newer models
    (and this DeepSeek-V4-Flash deployment) require ``max_completion_tokens`` and reject
    ``max_tokens``; some also fix ``temperature``. Try the variants in order and return
    the first that succeeds, so callers don't have to care which the endpoint wants.
    """
    client, model = judge.client, judge.model_name
    attempts = [
        {"max_completion_tokens": max_tokens, "temperature": temperature},
        {"max_tokens": max_tokens, "temperature": temperature},
        {"max_completion_tokens": max_tokens},
        {"max_tokens": max_tokens},
    ]
    last_err: Exception | None = None
    for kwargs in attempts:
        try:
            return client.chat.completions.create(model=model, messages=messages, **kwargs)
        except Exception as exc:  # try the next parameter spelling
            last_err = exc
    raise last_err  # type: ignore[misc]
