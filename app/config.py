"""Read application settings without storing credentials in source code."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import dotenv_values


class ConfigurationError(Exception):
    """Indicate that settings needed to answer a chat request are missing."""


@dataclass(frozen=True)
class Settings:
    """Hold validated settings; exclude the secret key from printed representations."""

    openai_api_key: str = field(repr=False)
    openai_model: str = "gpt-4.1-mini"


def get_settings() -> Settings:
    """Read the project .env file, letting environment variables take precedence."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    # Read on demand so /health works before configuration, and local .env edits
    # take effect on the next request without changing the process environment.
    values = {**dotenv_values(env_path), **os.environ}
    api_key = (values.get("OPENAI_API_KEY") or "").strip()
    model = (values.get("OPENAI_MODEL", "gpt-4.1-mini") or "").strip()

    if not api_key:
        raise ConfigurationError("Set OPENAI_API_KEY in your local .env file.")
    if not model:
        raise ConfigurationError("Set OPENAI_MODEL in your local .env file.")

    return Settings(openai_api_key=api_key, openai_model=model)
