"""Load parsed, chunked, embedded documentation into Postgres. Safe to re-run at any time.

Two passes:

1. Plan. Parse and chunk every page. A page's hash covers its chunks (ids, texts, metadata)
   and the embedding model, so a page is "changed" when its docs changed *or* when the parser,
   chunker or model changed what it produces. Pages whose hash matches the stored one are
   skipped entirely.
2. Embed, then write.
   - Embeddings are keyed by (model, content hash) in embedding_cache. Only texts missing from
     the cache are embedded, in batches, and each batch is committed as soon as it's done:
     an interrupted run loses at most one batch, and text shared between versions is
     embedded once.
   - Each changed page is then written in its own transaction: page row, chunks (vectors
     copied from the cache), and deletion of chunks that no longer exist. The page's new hash
     is stored in that same transaction, so a page is either fully updated or untouched.
   - On a full run (no --limit), pages that are gone from disk are deleted with their chunks.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from docs_qa import db, parse
from docs_qa.chunk import Chunk, Chunker
from docs_qa.embed import EMBEDDING_DIM, Embedder

_LOCK_KEY = 4_215_200_002  # "docs-qa ingest": one ingest at a time per database
EMBED_BATCH = 256


class IngestError(Exception):
    pass


@dataclass
class IngestReport:
    versions: list[str]
    pages_new: int = 0
    pages_changed: int = 0
    pages_unchanged: int = 0
    pages_removed: int = 0
    chunks_written: int = 0
    chunks_deleted: int = 0
    distinct_texts: int = 0  # distinct chunk texts on changed pages (one embedding each)
    embeddings_computed: int = 0
    embeddings_reused: int = 0  # found in embedding_cache from an earlier run or version
    seconds: dict[str, float] = field(default_factory=dict)  # stage -> seconds
    sizes: dict[str, str] = field(default_factory=dict)  # table/index -> human size


@dataclass(frozen=True)
class _PlannedPage:
    page: parse.Page
    chunks: list[Chunk]
    hash: str
    is_new: bool


def page_hash(page: parse.Page, chunks: list[Chunk], model: str) -> str:
    payload = {
        "model": model,
        "title": page.title,
        "release": page.release,
        "chunks": [[c.id, c.content_hash, c.url, list(c.trail), c.token_count] for c in chunks],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def vector_literal(vector: list[float]) -> str:
    """pgvector's text format, e.g. '[0.1,-0.2]'. Cast with ::vector in SQL."""
    return "[" + ",".join(repr(float(v)) for v in vector) + "]"


def _batches(items: list, size: int) -> Iterator[list]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def ingest(
    conn,
    *,
    raw_dir: Path,
    versions: Iterable[str],
    chunker: Chunker,
    embedder: Embedder,
    limit: int | None = None,
    batch_size: int = EMBED_BATCH,
    log: Callable[[str], None] = print,
) -> IngestReport:
    versions = list(versions)
    report = IngestReport(versions)
    if embedder.dim != EMBEDDING_DIM:
        raise IngestError(
            f"{embedder.model} makes {embedder.dim}-dimension vectors; the schema stores "
            f"{EMBEDDING_DIM}. A different model size needs a new migration."
        )
    todo = db.pending(conn, db.discover())
    conn.rollback()
    if todo:
        raise IngestError(
            f"the schema is out of date ({len(todo)} pending migrations); run `docs-qa db up`"
        )
    if not conn.execute("SELECT pg_try_advisory_lock(%s)", (_LOCK_KEY,)).fetchone()[0]:
        raise IngestError("another ingest is running against this database")
    conn.commit()
    try:
        planned, removed = _plan(conn, raw_dir, versions, chunker, embedder, limit, report)
        _embed_missing(conn, planned, embedder, batch_size, report, log)
        _write(conn, planned, removed, embedder, report, log)
        report.sizes = _sizes(conn)
    finally:
        conn.rollback()
        conn.execute("SELECT pg_advisory_unlock(%s)", (_LOCK_KEY,))
        conn.commit()
    return report


def _plan(conn, raw_dir, versions, chunker, embedder, limit, report):
    started = time.monotonic()
    rows = conn.execute(
        "SELECT version, path, content_hash FROM pages WHERE version = ANY(%s)", (versions,)
    ).fetchall()
    conn.rollback()
    stored = {(v, p): h for v, p, h in rows}

    planned: list[_PlannedPage] = []
    seen: set[tuple[str, str]] = set()
    for version in versions:
        names = list(parse.iter_page_names(raw_dir, version))
        if not names:
            raise IngestError(f"no downloaded pages for version {version} in {raw_dir}")
        for name in names[:limit]:
            seen.add((version, name))
            page = parse.parse_file(raw_dir, version, name)
            chunks = chunker.chunk_page(page)
            new_hash = page_hash(page, chunks, embedder.model)
            old_hash = stored.get((version, name))
            if old_hash == new_hash:
                report.pages_unchanged += 1
                continue
            planned.append(_PlannedPage(page, chunks, new_hash, is_new=old_hash is None))

    # Only a full run knows a page is really gone; with --limit most pages are simply unread.
    removed = [] if limit is not None else sorted(set(stored) - seen)
    report.seconds["parse and chunk"] = time.monotonic() - started
    return planned, removed


def _embed_missing(conn, planned, embedder, batch_size, report, log) -> None:
    started = time.monotonic()
    needed = {c.content_hash: c.text for p in planned for c in p.chunks}
    cached = {
        h
        for (h,) in conn.execute(
            "SELECT content_hash FROM embedding_cache WHERE model = %s AND content_hash = ANY(%s)",
            (embedder.model, list(needed)),
        )
    }
    conn.rollback()
    missing = [(h, text) for h, text in needed.items() if h not in cached]
    report.distinct_texts = len(needed)
    report.embeddings_reused = len(needed) - len(missing)
    if missing:
        log(f"embedding {len(missing)} new texts ({report.embeddings_reused} already cached)")
    for batch in _batches(missing, batch_size):
        vectors = embedder.embed([text for _, text in batch])
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO embedding_cache (model, content_hash, embedding) "
                "VALUES (%s, %s, %s::vector) ON CONFLICT DO NOTHING",
                [
                    (embedder.model, h, vector_literal(v))
                    for (h, _), v in zip(batch, vectors, strict=True)
                ],
            )
        conn.commit()  # keep finished batches even if the run is interrupted
        report.embeddings_computed += len(batch)
        log(f"  embedded {report.embeddings_computed}/{len(missing)}")
    report.seconds["embed"] = time.monotonic() - started


_UPSERT_PAGE = """
INSERT INTO pages (version, path, url, title, release, content_hash, ingested_at)
VALUES (%s, %s, %s, %s, %s, %s, now())
ON CONFLICT (version, path) DO UPDATE SET
    url = EXCLUDED.url, title = EXCLUDED.title, release = EXCLUDED.release,
    content_hash = EXCLUDED.content_hash, ingested_at = now()
"""

_UPSERT_CHUNK = """
INSERT INTO chunks (id, version, page_path, section_key, ordinal, anchor, section_path, url,
                    content, content_hash, token_count, embedding, embedding_model, updated_at)
SELECT %(id)s, %(version)s, %(page_path)s, %(section_key)s, %(ordinal)s, %(anchor)s,
       %(trail)s, %(url)s, %(text)s, %(content_hash)s, %(token_count)s,
       e.embedding, e.model, now()
FROM embedding_cache e
WHERE e.model = %(model)s AND e.content_hash = %(content_hash)s
ON CONFLICT (id) DO UPDATE SET
    section_key = EXCLUDED.section_key, ordinal = EXCLUDED.ordinal, anchor = EXCLUDED.anchor,
    section_path = EXCLUDED.section_path, url = EXCLUDED.url, content = EXCLUDED.content,
    content_hash = EXCLUDED.content_hash, token_count = EXCLUDED.token_count,
    embedding = EXCLUDED.embedding, embedding_model = EXCLUDED.embedding_model,
    updated_at = now()
"""


def _write(conn, planned, removed, embedder, report, log) -> None:
    started = time.monotonic()
    for n, item in enumerate(planned, 1):
        page, chunks = item.page, item.chunks
        with conn.transaction():
            conn.execute(
                _UPSERT_PAGE,
                (page.version, page.path, page.url, page.title, page.release, item.hash),
            )
            deleted = conn.execute(
                "DELETE FROM chunks WHERE version = %s AND page_path = %s AND id <> ALL(%s)",
                (page.version, page.path, [c.id for c in chunks]),
            )
            report.chunks_deleted += deleted.rowcount
            with conn.cursor() as cur:
                cur.executemany(
                    _UPSERT_CHUNK,
                    [
                        {
                            "id": c.id,
                            "version": c.version,
                            "page_path": c.page_path,
                            "section_key": c.section_key,
                            "ordinal": c.ordinal,
                            "anchor": c.anchor,
                            "trail": list(c.trail),
                            "url": c.url,
                            "text": c.text,
                            "content_hash": c.content_hash,
                            "token_count": c.token_count,
                            "model": embedder.model,
                        }
                        for c in chunks
                    ],
                )
            stored = conn.execute(
                "SELECT count(*) FROM chunks WHERE version = %s AND page_path = %s "
                "AND embedding_model = %s",
                (page.version, page.path, embedder.model),
            ).fetchone()[0]
            if stored != len(chunks):  # a vector was missing from the cache: roll back the page
                raise IngestError(f"{page.version}/{page.path}: stored {stored} of {len(chunks)}")
        conn.commit()
        report.chunks_written += len(chunks)
        if item.is_new:
            report.pages_new += 1
        else:
            report.pages_changed += 1
        if n % 200 == 0:
            log(f"  wrote {n}/{len(planned)} pages")

    for version, path in removed:
        with conn.transaction():
            count = conn.execute(
                "SELECT count(*) FROM chunks WHERE version = %s AND page_path = %s",
                (version, path),
            ).fetchone()[0]
            conn.execute("DELETE FROM pages WHERE version = %s AND path = %s", (version, path))
        conn.commit()
        report.chunks_deleted += count
        report.pages_removed += 1
    report.seconds["write"] = time.monotonic() - started


_SIZES = """
SELECT pg_size_pretty(pg_total_relation_size('pages')),
       pg_size_pretty(pg_total_relation_size('chunks')),
       pg_size_pretty(pg_relation_size('chunks_embedding_hnsw')),
       pg_size_pretty(pg_total_relation_size('embedding_cache'))
"""


def _sizes(conn) -> dict[str, str]:
    pages, chunks, hnsw, cache = conn.execute(_SIZES).fetchone()
    conn.rollback()
    return {
        "pages": pages,
        "chunks (with indexes)": chunks,
        "of which HNSW index": hnsw,
        "embedding_cache": cache,
    }
