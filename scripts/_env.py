"""Minimal .env loader for Flec scripts.

Reads the project-root .env file and injects any unset variables into
os.environ — identical behaviour to python-dotenv but with zero extra deps.

Usage (at the top of any script, before other imports that need the vars):

    from _env import load_dotenv
    load_dotenv()
"""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(env_file: Path | None = None) -> dict[str, str]:
    """Load .env into os.environ.  Already-set vars are NOT overwritten.

    Returns a dict of the vars that were actually injected this call.
    """
    if env_file is None:
        # Walk up from this file's location to find the project root .env
        here = Path(__file__).resolve().parent
        for candidate in [here, here.parent]:
            p = candidate / ".env"
            if p.exists():
                env_file = p
                break

    if env_file is None or not env_file.exists():
        return {}

    injected: dict[str, str] = {}
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, raw_value = line.partition("=")
        key = key.strip()
        # Strip inline comments and surrounding quotes
        value = raw_value.split("#", 1)[0].strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value
            injected[key] = value

    return injected
