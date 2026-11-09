"""`docs-qa`: the project's command line.

Commands:
- `uv run docs-qa db up`: apply pending schema migrations
- `uv run docs-qa db status`: show applied and pending migrations
"""

from __future__ import annotations

import argparse
import sys

from docs_qa import db
from docs_qa.config import ConfigError, database_url


def _db_up() -> int:
    with db.connect(database_url()) as conn:
        done = db.migrate(conn)
    if done:
        for m in done:
            print(f"applied {m.path.name}")
    else:
        print("schema is up to date, nothing to apply")
    return 0


def _db_status() -> int:
    migrations = db.discover()
    with db.connect(database_url()) as conn:
        done = db.applied(conn)
        todo = db.pending(conn, migrations)
    for m in migrations:
        print(f"{'applied' if m.version in done else 'pending'}  {m.path.name}")
    print(f"\n{len(done)} applied, {len(todo)} pending")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="docs-qa")
    groups = parser.add_subparsers(dest="group", required=True)
    db_parser = groups.add_parser("db", help="database schema")
    db_commands = db_parser.add_subparsers(dest="command", required=True)
    db_commands.add_parser("up", help="apply pending migrations")
    db_commands.add_parser("status", help="show applied and pending migrations")
    args = parser.parse_args(argv)

    try:
        if (args.group, args.command) == ("db", "up"):
            return _db_up()
        return _db_status()
    except (db.MigrationError, ConfigError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
