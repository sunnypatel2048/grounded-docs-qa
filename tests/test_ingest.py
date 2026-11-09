"""Ingestion tests against the test database, with fake vectors (HashEmbedder)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from conftest import require_pgvector

from docs_qa import db, ingest, tokens
from docs_qa.chunk import Chunker
from docs_qa.embed import HashEmbedder
from docs_qa.ingest import IngestError, vector_literal

FIXTURES = Path(__file__).parent / "fixtures" / "16"
PAGES = sorted(p.name for p in FIXTURES.glob("*.html"))
CHUNKER = Chunker(tokens.WhitespaceCounter(), 400)


@pytest.fixture
def conn(db_conn):
    require_pgvector(db_conn)
    db.migrate(db_conn)
    return db_conn


@pytest.fixture
def raw(tmp_path):
    """A raw-pages directory holding the fixture pages as version 16 (tests may add 17)."""
    shutil.copytree(FIXTURES, tmp_path / "16")
    return tmp_path


def run(conn, raw, versions=("16",), embedder=None, **kwargs):
    embedder = embedder or HashEmbedder()
    report = ingest.ingest(
        conn,
        raw_dir=raw,
        versions=versions,
        chunker=CHUNKER,
        embedder=embedder,
        log=lambda _m: None,
        **kwargs,
    )
    return report, embedder


def scalar(conn, sql, params=()):
    value = conn.execute(sql, params).fetchone()[0]
    conn.rollback()
    return value


def edit(path: Path, old: str, new: str) -> None:
    html = path.read_text(encoding="utf-8")
    assert old in html
    path.write_text(html.replace(old, new), encoding="utf-8")


def expected_chunks(raw, version="16"):
    from docs_qa import parse

    return [c for name in PAGES for c in CHUNKER.chunk_page(parse.parse_file(raw, version, name))]


# --- first run and re-runs ----------------------------------------------------------------


@pytest.mark.db
def test_first_run_stores_every_page_and_chunk(conn, raw):
    report, embedder = run(conn, raw)
    chunks = expected_chunks(raw)

    assert (report.pages_new, report.pages_changed, report.pages_unchanged) == (4, 0, 0)
    assert report.chunks_written == len(chunks)
    assert report.embeddings_computed == len({c.content_hash for c in chunks})
    assert scalar(conn, "SELECT count(*) FROM pages") == 4
    assert scalar(conn, "SELECT count(*) FROM chunks WHERE embedding IS NOT NULL") == len(chunks)

    row = conn.execute(
        "SELECT p.release, c.section_key, c.ordinal, c.section_path, c.url, c.embedding_model "
        "FROM chunks c JOIN pages p ON (p.version, p.path) = (c.version, c.page_path) "
        "WHERE c.id = '16/runtime-config-autovacuum.html#GUC-AUTOVACUUM-NAPTIME/0'"
    ).fetchone()
    assert row == (
        "16.15",
        "GUC-AUTOVACUUM-NAPTIME",
        0,
        ["Automatic Vacuuming", "autovacuum_naptime (integer)"],
        "https://www.postgresql.org/docs/16/runtime-config-autovacuum.html#GUC-AUTOVACUUM-NAPTIME",
        "hash-embedder",
    )


@pytest.mark.db
def test_second_run_changes_nothing(conn, raw):
    run(conn, raw)
    before = conn.execute("SELECT path, ingested_at FROM pages ORDER BY path").fetchall()
    report, embedder = run(conn, raw)

    assert embedder.calls == []
    assert (report.pages_unchanged, report.pages_new, report.pages_changed) == (4, 0, 0)
    assert report.chunks_written == report.chunks_deleted == 0
    assert conn.execute("SELECT path, ingested_at FROM pages ORDER BY path").fetchall() == before


@pytest.mark.db
def test_text_shared_by_two_versions_is_embedded_once(conn, raw):
    shutil.copytree(raw / "16", raw / "17")
    report, embedder = run(conn, raw, versions=("16", "17"))
    chunks = expected_chunks(raw)

    assert scalar(conn, "SELECT count(*) FROM chunks") == 2 * len(chunks)
    assert len(embedder.embedded) == len({c.content_hash for c in chunks})
    assert scalar(conn, "SELECT count(*) FROM embedding_cache") == len(embedder.embedded)


@pytest.mark.db
def test_a_new_version_reuses_cached_embeddings(conn, raw):
    run(conn, raw, versions=("16",))
    shutil.copytree(raw / "16", raw / "17")
    report, embedder = run(conn, raw, versions=("17",))
    assert embedder.calls == []
    assert report.embeddings_reused == report.distinct_texts > 0


@pytest.mark.db
def test_editing_one_paragraph_reembeds_only_that_chunk(conn, raw):
    run(conn, raw)
    page = raw / "16" / "runtime-config-autovacuum.html"
    edit(page, "The default is one minute", "The default is two minutes")
    report, embedder = run(conn, raw)

    assert (report.pages_changed, report.pages_unchanged) == (1, 3)
    assert len(embedder.embedded) == 1 and "two minutes" in embedder.embedded[0]
    content = scalar(
        conn,
        "SELECT content FROM chunks "
        "WHERE id = '16/runtime-config-autovacuum.html#GUC-AUTOVACUUM-NAPTIME/0'",
    )
    assert "two minutes" in content


@pytest.mark.db
def test_a_removed_section_loses_its_chunks(conn, raw):
    run(conn, raw)
    page = raw / "16" / "runtime-config-autovacuum.html"
    html = page.read_text(encoding="utf-8")
    start = html.index('<dt id="GUC-AUTOVACUUM-NAPTIME"')
    end = html.index("</dd>", start) + len("</dd>")
    page.write_text(html[:start] + html[end:], encoding="utf-8")
    report, _ = run(conn, raw)

    assert report.chunks_deleted == 1
    gone = "16/runtime-config-autovacuum.html#GUC-AUTOVACUUM-NAPTIME/0"
    assert scalar(conn, "SELECT count(*) FROM chunks WHERE id = %s", (gone,)) == 0


@pytest.mark.db
def test_a_removed_page_is_deleted_only_by_a_full_run(conn, raw):
    run(conn, raw)
    (raw / "16" / "functions-string.html").unlink()

    report, _ = run(conn, raw, limit=10)  # a trial run can't tell "gone" from "not read"
    assert report.pages_removed == 0
    assert scalar(conn, "SELECT count(*) FROM pages") == 4

    report, _ = run(conn, raw)
    assert report.pages_removed == 1 and report.chunks_deleted > 0
    assert scalar(conn, "SELECT count(*) FROM pages") == 3
    assert (
        scalar(conn, "SELECT count(*) FROM chunks WHERE page_path = 'functions-string.html'") == 0
    )


@pytest.mark.db
def test_an_interrupted_run_keeps_finished_embedding_batches(conn, raw):
    class FailsOnSecondBatch(HashEmbedder):
        def embed(self, texts):
            if len(self.calls) == 1:
                raise KeyboardInterrupt
            return super().embed(texts)

    with pytest.raises(KeyboardInterrupt):
        run(conn, raw, embedder=FailsOnSecondBatch(), batch_size=2)
    assert scalar(conn, "SELECT count(*) FROM embedding_cache") == 2
    assert scalar(conn, "SELECT count(*) FROM pages") == 0  # nothing half-written

    report, embedder = run(conn, raw, batch_size=2)
    assert report.embeddings_reused == 2
    assert report.embeddings_computed == report.distinct_texts - 2
    assert report.pages_new == 4


# --- refusals -----------------------------------------------------------------------------


@pytest.mark.db
def test_an_out_of_date_schema_is_refused(db_conn, raw):
    with pytest.raises(IngestError, match="docs-qa db up"):
        run(db_conn, raw)


@pytest.mark.db
def test_a_model_with_the_wrong_vector_size_is_refused(conn, raw):
    with pytest.raises(IngestError, match="384"):
        run(conn, raw, embedder=HashEmbedder(dim=768))


@pytest.mark.db
def test_two_ingests_at_once_are_refused(conn, raw, test_db_url):
    import psycopg

    with psycopg.connect(test_db_url, autocommit=True) as other:
        other.execute("SELECT pg_advisory_lock(%s)", (ingest._LOCK_KEY,))
        with pytest.raises(IngestError, match="another ingest"):
            run(conn, raw)


def test_vector_literal():
    assert vector_literal([0.5, -1.0, 2]) == "[0.5,-1.0,2.0]"
