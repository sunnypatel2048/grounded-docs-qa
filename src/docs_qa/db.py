"""Database connection and a small migration runner.

Schema changes live in numbered SQL files (sql/001_init.sql, sql/002_....sql). `migrate()`
applies the ones not yet recorded in the schema_migrations table, in order, inside one
transaction, so a failed migration leaves the database unchanged. Rules:

  - Files are named NNN_short_name.sql; numbers must be unique.
  - An applied migration is never edited: its checksum is stored, and a changed file is an
    error. Change the schema by adding a new file.
  - An advisory lock stops two runners (two terminals, or CI jobs) applying at once.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # psycopg is imported lazily so pure-logic code and tests don't need it
    import psycopg

MIGRATIONS_DIR = Path(__file__).parent / "sql"
_FILENAME = re.compile(r"^(\d{3})_([a-z0-9_]+)\.sql$")
_LOCK_KEY = 4_215_200_001  # arbitrary constant identifying "docs-qa migrations"


class MigrationError(Exception):
    pass


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    path: Path

    @cached_property
    def sql(self) -> str:
        return self.path.read_text()

    @cached_property
    def checksum(self) -> str:
        return hashlib.sha256(self.sql.encode()).hexdigest()


def discover(directory: Path = MIGRATIONS_DIR) -> list[Migration]:
    migrations = []
    for path in sorted(directory.glob("*.sql")):
        match = _FILENAME.match(path.name)
        if not match:
            raise MigrationError(f"{path.name}: migration files must be named NNN_name.sql")
        migrations.append(Migration(int(match.group(1)), match.group(2), path))
    versions = [m.version for m in migrations]
    duplicates = sorted({v for v in versions if versions.count(v) > 1})
    if duplicates:
        raise MigrationError(f"duplicate migration numbers: {duplicates}")
    return sorted(migrations, key=lambda m: m.version)


def connect(url: str) -> psycopg.Connection:
    import psycopg

    return psycopg.connect(url)


_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version    integer     PRIMARY KEY,
    name       text        NOT NULL,
    checksum   text        NOT NULL,
    applied_at timestamptz NOT NULL DEFAULT now()
)
"""


def applied(conn: psycopg.Connection) -> dict[int, tuple[str, str]]:
    """{version: (name, checksum)} for every applied migration; empty if none yet."""
    exists = conn.execute("SELECT to_regclass('schema_migrations') IS NOT NULL").fetchone()[0]
    if not exists:
        return {}
    rows = conn.execute("SELECT version, name, checksum FROM schema_migrations").fetchall()
    return {version: (name, checksum) for version, name, checksum in rows}


def pending(conn: psycopg.Connection, migrations: list[Migration]) -> list[Migration]:
    """Migrations not yet applied. Raises if an applied file was edited or has disappeared."""
    done = applied(conn)
    known = {m.version: m for m in migrations}
    for version, (name, checksum) in sorted(done.items()):
        migration = known.get(version)
        if migration is None:
            raise MigrationError(f"migration {version:03d}_{name} was applied but its file is gone")
        if migration.checksum != checksum:
            raise MigrationError(
                f"{migration.path.name} was edited after it was applied. "
                "Revert it and put the change in a new migration file."
            )
    return [m for m in migrations if m.version not in done]


def migrate(conn: psycopg.Connection, migrations: list[Migration] | None = None) -> list[Migration]:
    """Apply pending migrations in one transaction. Returns what was applied.."""
    migrations = discover() if migrations is None else migrations
    with conn.transaction():
        conn.execute("SELECT pg_advisory_xact_lock(%s)", (_LOCK_KEY,))
        conn.execute(_CREATE_TABLE)
        todo = pending(conn, migrations)
        for migration in todo:
            conn.execute(migration.sql)
            conn.execute(
                "INSERT INTO schema_migrations (version, name, checksum) VALUES (%s, %s, %s)",
                (migration.version, migration.name, migration.checksum),
            )
    return todo
