import os

import pytest

from docs_qa import config


@pytest.fixture
def test_db_url() -> str:
    """TEST_DATABASE_URL from the environment or .env. Skips if unset; fails if it could be the
    real database, because the tests wipe it."""
    url = config.test_database_url()
    if url is None:
        pytest.skip("TEST_DATABASE_URL isn't set (see .env.example)")
    try:
        config.check_test_database(url, os.environ.get("DATABASE_URL"))
    except config.ConfigError as e:
        pytest.fail(str(e))
    return url


@pytest.fixture
def db_conn(test_db_url):
    """A connection to an empty test database. Skips the test if it isn't reachable.

    Wipes the public schema first, so every test starts clean.
    """
    psycopg = pytest.importorskip("psycopg")
    try:
        conn = psycopg.connect(test_db_url, connect_timeout=2)
    except psycopg.OperationalError as e:
        pytest.skip(f"no test database reachable ({e.__class__.__name__})")
    conn.autocommit = True
    conn.execute("DROP SCHEMA public CASCADE")
    conn.execute("CREATE SCHEMA public")
    conn.autocommit = False
    yield conn
    conn.close()


def require_pgvector(conn) -> None:
    """Skip unless the vector extension is installed (it is in the pgvector image and CI)."""
    available = conn.execute(
        "SELECT count(*) FROM pg_available_extensions WHERE name = 'vector'"
    ).fetchone()[0]
    conn.rollback()
    if not available:
        pytest.skip("pgvector isn't installed in this Postgres")
