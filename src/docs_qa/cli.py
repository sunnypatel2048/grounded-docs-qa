"""`docs-qa`: the project's command line.

Commands:
- `uv run docs-qa db up`: apply pending schema migrations
- `uv run docs-qa db status`: show applied and pending migrations
- `uv run docs-qa fetch --versions 16 17 18`: download the documentation pages (resumable)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from docs_qa import db, fetch
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


def _fetch(versions: list[str], delay: float, limit: int | None, raw_dir: Path) -> int:
    crawler = fetch.Crawler(raw_dir=raw_dir, delay=delay)
    print(f"one request every {crawler.delay:g}s; pages go to {raw_dir}/<version>/")
    failed = 0
    for version in versions:
        print(f"{version}: crawling")
        r = crawler.crawl(version, limit=limit)
        print(
            f"{version}: {r.pages} pages ({r.downloaded} downloaded, {r.cached} already on disk), "
            f"{r.skipped_by_robots} skipped by robots.txt, {len(r.failed)} failed, "
            f"{r.bytes_downloaded / 1_000_000:.1f} MB in {r.seconds:.0f}s"
        )
        for url in r.failed:
            print(f"  failed: {url}")
        failed += len(r.failed)
    if failed:
        print(f"{failed} pages failed; run the same command again to retry them")
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="docs-qa")
    groups = parser.add_subparsers(dest="group", required=True)
    db_parser = groups.add_parser("db", help="database schema")
    db_commands = db_parser.add_subparsers(dest="command", required=True)
    db_commands.add_parser("up", help="apply pending migrations")
    db_commands.add_parser("status", help="show applied and pending migrations")

    fetch_parser = groups.add_parser("fetch", help="download the documentation pages")
    fetch_parser.add_argument("--versions", nargs="+", default=["16", "17", "18"])
    fetch_parser.add_argument(
        "--delay", type=float, default=1.0, help="seconds between requests (default 1.0)"
    )
    fetch_parser.add_argument(
        "--limit", type=int, default=None, help="stop after N pages per version (for a trial)"
    )
    fetch_parser.add_argument("--raw-dir", type=Path, default=fetch.DEFAULT_RAW_DIR)
    args = parser.parse_args(argv)

    try:
        if args.group == "fetch":
            return _fetch(args.versions, args.delay, args.limit, args.raw_dir)
        if args.command == "up":
            return _db_up()
        return _db_status()
    except (db.MigrationError, ConfigError, fetch.CrawlAborted) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted; run the same command again to resume", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
