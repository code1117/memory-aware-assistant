"""Read application settings without storing credentials in source code."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import dotenv_values


class ConfigurationError(Exception):
    """Indicate that required application settings are missing or invalid."""


@dataclass(frozen=True)
class Settings:
    """Hold validated settings; exclude the secret key from printed representations."""

    openai_api_key: str = field(repr=False)
    openai_model: str = "gpt-4.1-mini"


@dataclass(frozen=True)
class Neo4jSettings:
    """Hold database connection settings without displaying the password."""

    uri: str
    username: str
    password: str = field(repr=False)
    database: str


def _read_environment() -> dict[str, str | None]:
    """Read local settings, letting process environment variables take precedence."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    # Read on demand so /health works before configuration, and local .env edits
    # take effect on the next request without changing the process environment.
    return {**dotenv_values(env_path), **os.environ}


def get_settings() -> Settings:
    """Validate OpenAI settings independently of the database configuration."""
    values = _read_environment()
    api_key = (values.get("OPENAI_API_KEY") or "").strip()
    model = (values.get("OPENAI_MODEL", "gpt-4.1-mini") or "").strip()

    if not api_key:
        raise ConfigurationError("Set OPENAI_API_KEY in your local .env file.")
    if not model:
        raise ConfigurationError("Set OPENAI_MODEL in your local .env file.")

    return Settings(openai_api_key=api_key, openai_model=model)


def get_neo4j_settings() -> Neo4jSettings:
    """Validate database settings without requiring an OpenAI API key."""
    values = _read_environment()
    uri = (values.get("NEO4J_URI", "bolt://localhost:7687") or "").strip()
    username = (values.get("NEO4J_USERNAME", "neo4j") or "").strip()
    database = (values.get("NEO4J_DATABASE", "neo4j") or "").strip()
    # Preserve the actual password, including any intentional whitespace.
    password = values.get("NEO4J_PASSWORD") or ""

    if not username:
        raise ConfigurationError("Set NEO4J_USERNAME in your local .env file.")
    if not password.strip():
        raise ConfigurationError("Set NEO4J_PASSWORD in your local .env file.")
    if not database:
        raise ConfigurationError("Set NEO4J_DATABASE in your local .env file.")

    # Accept the driver's Bolt/Neo4j schemes, but keep credentials in their
    # dedicated settings rather than embedding them in the connection address.
    try:
        address = urlsplit(uri)
        valid_address = (
            address.scheme in {
                "bolt", "bolt+s", "bolt+ssc", "neo4j", "neo4j+s", "neo4j+ssc"
            }
            and bool(address.hostname)
            and (address.port is None or 1 <= address.port <= 65535)
            and address.username is None
            and address.password is None
            and not address.path
            and not address.query
            and not address.fragment
        )
    except ValueError:
        valid_address = False
    if not valid_address:
        raise ConfigurationError(
            "Set NEO4J_URI to a database address such as bolt://localhost:7687."
        )

    return Neo4jSettings(uri=uri, username=username, password=password, database=database)
