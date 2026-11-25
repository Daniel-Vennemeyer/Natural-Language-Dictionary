"""API key lookup for OpenAI-compatible clients."""
from __future__ import annotations

import os


def _load_key(env_path: str = ".env", var: str = "OPENAI_API_KEY") -> str:
    """Return `var` from the environment, else from a .env file (env_path, ./.env, ~/.env)."""
    if os.environ.get(var):
        return os.environ[var]
    for p in (env_path, ".env", os.path.expanduser("~/.env")):
        if p and os.path.exists(p):
            for line in open(p):
                line = line.strip()
                if line.startswith(var + "=") and not line.startswith("#"):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError(f"{var} not found in environment or .env")
