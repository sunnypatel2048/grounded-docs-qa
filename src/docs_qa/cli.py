"""`docs-qa`: the project's command line.

Commands:
- `uv run docs-qa db up`: apply pending schema migrations
- `uv run docs-qa db status`: show applied and pending migrations
- `uv run docs-qa fetch --versions 16 17 18`: download the documentation pages (resumable)
- `uv run docs-qa parse 16/sql-select.html`: show how a page splits into sections and blocks
- `uv run docs-qa parse --all`: parse every downloaded page and print totals
- `uv run docs-qa chunk 16/sql-select.html`: show a page's chunks
- `uv run docs-qa chunk --report`: chunk every page and print sizes and embedding savings
- `uv run docs-qa ingest`: parse, chunk, embed and store everything (re-runs skip unchanged pages)
- `uv run docs-qa golden check`: validate eval/golden.yaml against its rules and the stored docs
- `uv run docs-qa golden sample`: random sections, spread across doc areas, to write questions from
- `uv run docs-qa golden changes --from 16 --to 17`: sections that changed, or are new, in a version
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from docs_qa import chunk, db, embed, explore, fetch, golden, ingest, parse, tokens
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
    versions = _downloaded_versions(raw_dir)
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


def _downloaded_versions(raw_dir: Path) -> list[str]:
    return sorted(p.name for p in raw_dir.glob("*") if p.is_dir() and p.name.isdigit())


def _percentile(sorted_values: list[int], p: float) -> int:
    return sorted_values[round(p * (len(sorted_values) - 1))]


def _chunk_show(pages: list[str], raw_dir: Path, chunker: chunk.Chunker) -> int:
    for ref in pages:
        version, _, name = ref.partition("/")
        if not (raw_dir / version / name).is_file():
            print(
                f"error: {raw_dir / ref} not found (expected e.g. 16/sql-select.html)",
                file=sys.stderr,
            )
            return 1
        for c in chunker.chunk_page(parse.parse_file(raw_dir, version, name)):
            print(f"=== {c.id}  ({c.token_count} tokens)")
            print(f"    {c.url}")
            print(c.text)
            print()
    return 0


def _chunk_report(raw_dir: Path, chunker: chunk.Chunker) -> int:
    versions = _downloaded_versions(raw_dir)
    if not versions:
        print(
            f"error: no downloaded pages in {raw_dir}; run `docs-qa fetch` first", file=sys.stderr
        )
        return 1
    started = time.monotonic()
    hashes: set[str] = set()
    total = 0
    print(f"tokenizer: {chunker.counter.model}, at most {chunker.max_tokens} tokens per chunk\n")
    for version in versions:
        sizes: list[int] = []
        for name in parse.iter_page_names(raw_dir, version):
            for c in chunker.chunk_page(parse.parse_file(raw_dir, version, name)):
                sizes.append(c.token_count)
                hashes.add(c.content_hash)
        sizes.sort()
        total += len(sizes)
        print(
            f"{version}: {len(sizes)} chunks; tokens p50 {_percentile(sizes, 0.5)}, "
            f"p90 {_percentile(sizes, 0.9)}, p99 {_percentile(sizes, 0.99)}, max {sizes[-1]}"
        )
    s = chunker.stats
    print(
        f"\noversized blocks split: {s.split_code} code (between lines), "
        f"{s.split_prose} prose (between sentences); {s.token_cuts} cut at token boundaries"
    )
    print(
        f"distinct chunk texts: {len(hashes)} of {total} "
        f"({1 - len(hashes) / total:.0%} need no embedding of their own)"
    )
    print(f"took {time.monotonic() - started:.0f}s")
    return 0


def _ingest(versions: list[str], limit: int | None, raw_dir: Path) -> int:
    try:
        counter = tokens.HFTokenCounter()
        embedder = embed.LocalEmbedder()
    except ImportError as e:
        print(
            f"error: {e.name} isn't installed. It's in the `embed` dependency group: "
            "run `uv sync` (or `uv sync --group embed`).",
            file=sys.stderr,
        )
        return 1
    print(f"embedding with {embedder.model} on {embedder.device}")
    with db.connect(database_url()) as conn:
        r = ingest.ingest(
            conn,
            raw_dir=raw_dir,
            versions=versions,
            chunker=chunk.Chunker(counter),
            embedder=embedder,
            limit=limit,
        )
    print(
        f"\npages: {r.pages_new} new, {r.pages_changed} changed, "
        f"{r.pages_unchanged} unchanged, {r.pages_removed} removed"
    )
    print(f"chunks: {r.chunks_written} written, {r.chunks_deleted} deleted")
    print(
        f"embeddings: {r.distinct_texts} distinct texts; {r.embeddings_computed} computed, "
        f"{r.embeddings_reused} reused from the cache"
    )
    print("time: " + ", ".join(f"{stage} {s:.0f}s" for stage, s in r.seconds.items()))
    print("storage: " + ", ".join(f"{name} {size}" for name, size in r.sizes.items()))
    return 0


def _golden_check(path: Path, offline: bool) -> int:
    gold = golden.load(path)
    problems = list(gold.problems)
    if not offline:
        with db.connect(database_url()) as conn:
            problems += golden.evidence_problems(conn, gold)
    for line in golden.mix_report(gold):
        print(line)
    if problems:
        print(f"\n{len(problems)} problems:")
        for p in problems:
            print(f"  {p}")
        return 1
    checked = "rules only (--offline)" if offline else "rules and evidence quotes"
    print(f"\nOK: {checked}")
    return 0


def _clip(text: str, limit: int) -> str:
    return (
        text
        if len(text) <= limit
        else text[:limit].rstrip() + f" … ({len(text) - limit} more characters)"
    )


def _indent(text: str, prefix: str = "   ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())


def _used_refs(path: Path) -> set[str]:
    """Evidence refs already in the golden set, so they aren't offered again."""
    if not path.exists():
        return set()
    return {e.ref for q in golden.load(path).questions for e in q.evidence}


def _golden_sample(versions: list[str], n: int, seed: int, area: str | None, path: Path) -> int:
    with db.connect(database_url()) as conn:
        sections = explore.load_sections(conn, versions)
    if area:
        sections = [s for s in sections if explore.area_for(s.page) == area]
    picked = explore.sample(sections, n, seed, exclude=_used_refs(path))
    if not picked:
        print("no sections to sample; is the database ingested?", file=sys.stderr)
        return 1
    for s in picked:
        print(f"─── {s.ref}   (area: {explore.area_for(s.page)}, {s.tokens} tokens)")
        print(f"   {' › '.join(s.trail)}")
        print(f"   {s.url}\n")
        print(_indent(_clip(s.text, 2000)))
        print()
    print(f"seed {seed}: run with --seed {seed + 1} for a different set")
    return 0


def _golden_changes(old: str, new: str, n: int, seed: int, show_new: bool, path: Path) -> int:
    import random

    with db.connect(database_url()) as conn:
        sections = explore.load_sections(conn, [old, new])
    found, added = explore.changes(
        [s for s in sections if s.version == old], [s for s in sections if s.version == new]
    )
    used = _used_refs(path)
    rng = random.Random(seed)
    if show_new:
        pool = [s for s in added if s.ref not in used]
        print(
            f"{len(pool)} sections are new in {new} (not in {old}); showing {min(n, len(pool))}\n"
        )
        for s in sorted(rng.sample(pool, min(n, len(pool))), key=lambda s: s.ref):
            print(f"─── new in {new}: {s.ref}")
            print(f"   {' › '.join(s.trail)}\n")
            print(_indent(_clip(s.text, 600)))
            print()
    else:
        pool = [c for c in found if c.before.ref not in used and c.after.ref not in used]
        print(f"{len(pool)} sections changed from {old} to {new}; showing {min(n, len(pool))}\n")
        for c in sorted(rng.sample(pool, min(n, len(pool))), key=lambda c: c.after.ref):
            print(f"─── changed: {c.before.ref}")
            if c.after.page != c.before.page:
                print(f"         now: {c.after.ref}")
            print(f"   {' › '.join(c.after.trail)}   ({c.changed_words} words changed)\n")
            for sentence in c.removed[:8]:
                print(_indent(_clip(sentence, 300), f"   - {old}: "))
            for sentence in c.added[:8]:
                print(_indent(_clip(sentence, 300), f"   + {new}: "))
            print()
    print(f"seed {seed}: run with --seed {seed + 1} for a different set")
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

    parse_parser = groups.add_parser("parse", help="show how pages split into sections")
    parse_parser.add_argument("pages", nargs="*", help="pages to show, e.g. 16/sql-select.html")
    parse_parser.add_argument(
        "--all", action="store_true", help="parse every downloaded page and print totals"
    )
    parse_parser.add_argument("--raw-dir", type=Path, default=fetch.DEFAULT_RAW_DIR)

    chunk_parser = groups.add_parser("chunk", help="show how pages split into chunks")
    chunk_parser.add_argument("pages", nargs="*", help="pages to show, e.g. 16/sql-select.html")
    chunk_parser.add_argument(
        "--report", action="store_true", help="chunk every page and print totals"
    )
    chunk_parser.add_argument("--max-tokens", type=int, default=chunk.MAX_TOKENS)
    chunk_parser.add_argument("--raw-dir", type=Path, default=fetch.DEFAULT_RAW_DIR)

    ingest_parser = groups.add_parser("ingest", help="parse, chunk, embed and store the docs")
    ingest_parser.add_argument("--versions", nargs="+", default=["16", "17", "18"])
    ingest_parser.add_argument(
        "--limit", type=int, default=None, help="first N pages per version only (for a trial)"
    )
    ingest_parser.add_argument("--raw-dir", type=Path, default=fetch.DEFAULT_RAW_DIR)
    golden_parser = groups.add_parser("golden", help="the evaluation golden set")
    golden_commands = golden_parser.add_subparsers(dest="command", required=True)
    check_parser = golden_commands.add_parser("check", help="validate the golden set")
    check_parser.add_argument("--file", type=Path, default=golden.DEFAULT_PATH)
    check_parser.add_argument(
        "--offline", action="store_true", help="check the rules only, not the stored docs"
    )
    sample_parser = golden_commands.add_parser("sample", help="random sections to write from")
    sample_parser.add_argument("-n", type=int, default=10)
    sample_parser.add_argument("--versions", nargs="+", default=["16", "17", "18"])
    sample_parser.add_argument("--seed", type=int, default=0)
    sample_parser.add_argument("--area", choices=[*golden.AREAS, "other"])
    sample_parser.add_argument("--file", type=Path, default=golden.DEFAULT_PATH)
    changes_parser = golden_commands.add_parser(
        "changes", help="sections that changed or are new between two versions"
    )
    changes_parser.add_argument("--from", dest="old", required=True)
    changes_parser.add_argument("--to", dest="new", required=True)
    changes_parser.add_argument("-n", type=int, default=10)
    changes_parser.add_argument("--seed", type=int, default=0)
    changes_parser.add_argument(
        "--new",
        dest="show_new",
        action="store_true",
        help="sections new in --to, instead of changed ones",
    )
    changes_parser.add_argument("--file", type=Path, default=golden.DEFAULT_PATH)
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
        if args.group == "golden":
            if args.command == "sample":
                return _golden_sample(args.versions, args.n, args.seed, args.area, args.file)
            if args.command == "changes":
                return _golden_changes(
                    args.old, args.new, args.n, args.seed, args.show_new, args.file
                )
            return _golden_check(args.file, args.offline)
        if args.group == "ingest":
            return _ingest(args.versions, args.limit, args.raw_dir)
        if args.group == "chunk":
            if not args.report and not args.pages:
                parser.error("chunk: give one or more pages (e.g. 16/sql-select.html) or --report")
            if args.max_tokens < 50:
                parser.error("chunk: --max-tokens must be at least 50")
            try:
                counter = tokens.HFTokenCounter()
            except Exception as e:  # missing package, no network on first download, ...
                print(
                    f"error: couldn't load the {tokens.EMBEDDING_MODEL} tokenizer: {e}",
                    file=sys.stderr,
                )
                return 1
            chunker = chunk.Chunker(counter, args.max_tokens)
            if args.report:
                return _chunk_report(args.raw_dir, chunker)
            return _chunk_show(args.pages, args.raw_dir, chunker)
        if args.command == "up":
            return _db_up()
        return _db_status()
    except (
        db.MigrationError,
        ConfigError,
        fetch.CrawlAborted,
        parse.ParseError,
        ingest.IngestError,
        golden.GoldenError,
    ) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted; run the same command again to resume", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
