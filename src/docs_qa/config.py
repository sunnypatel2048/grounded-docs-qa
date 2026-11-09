"""Settings, read from the environment and from .env.

There are no hardcoded fallbacks: .env.example and docker-compose.yml document the values, and a
missing setting is an error with a fix, not a silent guess at where the database is.
"""

from __future__ import annotations

import os
from urllib.parse import urlparse

from dotenv import find_dotenv, load_dotenv


class ConfigError(Exception):
    pass


def _load_env() -> None:
    # Search for .env from the current directory upward. Variables already set in the shell win.
    load_dotenv(find_dotenv(usecwd=True), override=False)


def database_url() -> str:
    _load_env()
    url = os.environ.get("DATABASE_URL", "").strip()
    if not url:
        raise ConfigError(
            "DATABASE_URL isn't set. Copy .env.example to .env, or export DATABASE_URL."
        )
    return url


def test_database_url() -> str | None:
    """The database the tests may wipe, or None if not configured."""
    _load_env()
    return os.environ.get("TEST_DATABASE_URL", "").strip() or None


def database_name(url: str) -> str:
    """'postgresql://user:pw@host:5433/docsqa_test' -> 'docsqa_test'."""
    return urlparse(url).path.lstrip("/")


def check_test_database(test_url: str, app_url: str | None) -> None:
    """Refuse a test database that could be the real one. The tests drop the whole schema,
    so a typo in .env must fail loudly instead of wiping the ingested docs."""
    name = database_name(test_url)
    if not name.endswith("_test"):
        raise ConfigError(
            f"TEST_DATABASE_URL points at database {name!r}. The tests wipe it, so its name "
            "must end in '_test'."
        )
    if app_url is not None and database_name(app_url) == name:
        raise ConfigError(
            f"TEST_DATABASE_URL and DATABASE_URL both point at {name!r}. "
            "The tests need their own database."
        )
