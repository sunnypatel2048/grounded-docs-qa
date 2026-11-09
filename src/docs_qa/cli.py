"""`docs-qa`: the project's command line.

Commands:
- `uv run docs-qa db up`: apply pending schema migrations
- `uv run docs-qa db status`: show applied and pending migrations
- `uv run docs-qa fetch --versions 16 17 18`: download the documentation pages (resumable)
- `uv run docs-qa parse 16/sql-select.html`: show how a page splits into sections and blocks
- `uv run docs-qa parse --all`: parse every downloaded page and print totals
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from docs_qa import db, fetch, parse
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


def _parse_show(pages: list[str], raw_dir: Path) -> int:
    for ref in pages:
        version, _, name = ref.partition("/")
        if not (raw_dir / version / name).is_file():
            print(
                f"error: {raw_dir / ref} not found (expected e.g. 16/sql-select.html)",
                file=sys.stderr,
            )
            return 1
        page = parse.parse_file(raw_dir, version, name)
        print(f"{page.title}  [{page.kind}, release {page.release}]  {page.url}")
        print(f"{len(page.sections)} sections\n")
        for s in page.sections:
            print(f"## {' > '.join(s.trail)}")
            print(f"   key: {s.key}")
            print(f"   url: {s.url}")
            for block in s.blocks:
                print("   | " + block.replace("\n", "\n   | "))
                print("   |")
            print()
    return 0


def _parse_all(raw_dir: Path) -> int:
    versions = sorted(p.name for p in raw_dir.glob("*") if p.is_dir() and p.name.isdigit())
    if not versions:
        print(
            f"error: no downloaded pages in {raw_dir}; run `docs-qa fetch` first", file=sys.stderr
        )
        return 1
    errors: list[str] = []
    for version in versions:
        pages = sections = blocks = empty = 0
        for name in parse.iter_page_names(raw_dir, version):
            try:
                page = parse.parse_file(raw_dir, version, name)
            except parse.ParseError as e:
                errors.append(str(e))
                continue
            pages += 1
            sections += len(page.sections)
            blocks += sum(len(s.blocks) for s in page.sections)
            empty += not page.sections
        print(
            f"{version}: {pages} pages, {sections} sections, {blocks} blocks; "
            f"{empty} pages with no content (tables of contents, the index)"
        )
    for error in errors:
        print(f"  failed: {error}")
    return 1 if errors else 0


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

    parse_parser = groups.add_parser("parse", help="show how pages split into sections")
    parse_parser.add_argument("pages", nargs="*", help="pages to show, e.g. 16/sql-select.html")
    parse_parser.add_argument(
        "--all", action="store_true", help="parse every downloaded page and print totals"
    )
    parse_parser.add_argument("--raw-dir", type=Path, default=fetch.DEFAULT_RAW_DIR)
    args = parser.parse_args(argv)

    try:
        if args.group == "fetch":
            return _fetch(args.versions, args.delay, args.limit, args.raw_dir)
        if args.group == "parse":
            if args.all:
                return _parse_all(args.raw_dir)
            if not args.pages:
                parser.error("parse: give one or more pages (e.g. 16/sql-select.html) or --all")
            return _parse_show(args.pages, args.raw_dir)
        if args.command == "up":
            return _db_up()
        return _db_status()
    except (db.MigrationError, ConfigError, fetch.CrawlAborted, parse.ParseError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted; run the same command again to resume", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
