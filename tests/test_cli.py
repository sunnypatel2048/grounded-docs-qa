from pathlib import Path

import pytest
from conftest import require_pgvector

from docs_qa import cli, embed, fetch, tokens


def test_a_command_group_is_required():
    with pytest.raises(SystemExit):
        cli.main([])


def test_unknown_subcommand_is_rejected():
    with pytest.raises(SystemExit):
        cli.main(["db", "drop-everything"])


@pytest.mark.db
def test_db_up_twice_then_status(db_conn, test_db_url, monkeypatch, capsys):
    require_pgvector(db_conn)
    db_conn.close()  # the command opens its own connection
    monkeypatch.setenv("DATABASE_URL", test_db_url)

    assert cli.main(["db", "up"]) == 0
    assert "applied 001_init.sql" in capsys.readouterr().out

    assert cli.main(["db", "up"]) == 0
    assert "nothing to apply" in capsys.readouterr().out

    assert cli.main(["db", "status"]) == 0
    out = capsys.readouterr().out
    assert "applied  001_init.sql" in out and "2 applied, 0 pending" in out


def test_fetch_help_lists_options(capsys):
    with pytest.raises(SystemExit):
        cli.main(["fetch", "--help"])
    out = capsys.readouterr().out
    for option in ("--versions", "--delay", "--limit", "--raw-dir"):
        assert option in out


def _fake_crawler(site, time):
    real = fetch.Crawler

    def make(**kwargs):
        return real(fetch=site, sleep=time.sleep, clock=time.clock, log=lambda _m: None, **kwargs)

    return make


def test_fetch_reports_each_version(monkeypatch, tmp_path, capsys):
    from test_fetch import SITE, FakeSite, FakeTime

    monkeypatch.setattr(fetch, "Crawler", _fake_crawler(FakeSite(SITE), FakeTime()))
    code = cli.main(["fetch", "--versions", "16", "--raw-dir", str(tmp_path)])

    assert code == 0
    assert "16: 3 pages (3 downloaded, 0 already on disk)" in capsys.readouterr().out


def test_fetch_exits_1_when_pages_failed(monkeypatch, tmp_path, capsys):
    from test_fetch import DOCS, SITE, FakeSite, FakeTime

    site = FakeSite(SITE)
    site.respond(DOCS + "sql.html", fetch.Response(404, b"", {}))
    monkeypatch.setattr(fetch, "Crawler", _fake_crawler(site, FakeTime()))
    code = cli.main(["fetch", "--versions", "16", "--raw-dir", str(tmp_path)])

    assert code == 1
    assert "run the same command again" in capsys.readouterr().out


def test_fetch_aborted_is_a_clean_error(monkeypatch, tmp_path, capsys):
    from test_fetch import SITE, FakeSite, FakeTime

    site = FakeSite(SITE, robots=fetch.Response(503, b"", {}))
    monkeypatch.setattr(fetch, "Crawler", _fake_crawler(site, FakeTime()))
    code = cli.main(["fetch", "--raw-dir", str(tmp_path)])

    assert code == 1
    assert "error: couldn't read robots.txt" in capsys.readouterr().err


FIXTURES = str(Path(__file__).parent / "fixtures")


def test_parse_shows_a_pages_sections(capsys):
    code = cli.main(["parse", "16/runtime-config-autovacuum.html", "--raw-dir", FIXTURES])
    out = capsys.readouterr().out
    assert code == 0
    assert "## Automatic Vacuuming > autovacuum_naptime (integer)" in out
    assert "key: GUC-AUTOVACUUM-NAPTIME" in out


def test_parse_missing_page_is_a_clean_error(capsys):
    code = cli.main(["parse", "16/nope.html", "--raw-dir", FIXTURES])
    assert code == 1
    assert "not found" in capsys.readouterr().err


def test_parse_all_prints_totals(capsys):
    code = cli.main(["parse", "--all", "--raw-dir", FIXTURES])
    assert code == 0
    assert "16: 4 pages, 14 sections" in capsys.readouterr().out


def test_parse_all_without_downloads_says_to_fetch(tmp_path, capsys):
    assert cli.main(["parse", "--all", "--raw-dir", str(tmp_path)]) == 1
    assert "docs-qa fetch" in capsys.readouterr().err


def test_chunk_shows_a_pages_chunks(monkeypatch, capsys):
    monkeypatch.setattr(tokens, "HFTokenCounter", tokens.WhitespaceCounter)
    code = cli.main(["chunk", "16/runtime-config-autovacuum.html", "--raw-dir", FIXTURES])
    out = capsys.readouterr().out
    assert code == 0
    assert "=== 16/runtime-config-autovacuum.html#GUC-AUTOVACUUM-NAPTIME/0" in out


def test_chunk_report_prints_totals(monkeypatch, capsys):
    monkeypatch.setattr(tokens, "HFTokenCounter", tokens.WhitespaceCounter)
    code = cli.main(["chunk", "--report", "--raw-dir", FIXTURES])
    out = capsys.readouterr().out
    assert code == 0
    assert "16: " in out and "distinct chunk texts:" in out


def test_chunk_without_tokenizer_is_a_clean_error(monkeypatch, capsys):
    def unavailable():
        raise OSError("no network")

    monkeypatch.setattr(tokens, "HFTokenCounter", unavailable)
    code = cli.main(["chunk", "--report", "--raw-dir", FIXTURES])
    assert code == 1
    assert "couldn't load the BAAI/bge-small-en-v1.5 tokenizer" in capsys.readouterr().err


@pytest.mark.db
def test_ingest_prints_a_report(db_conn, test_db_url, monkeypatch, capsys):
    require_pgvector(db_conn)
    db_conn.close()

    class FakeEmbedder(embed.HashEmbedder):
        device = "cpu"

    monkeypatch.setenv("DATABASE_URL", test_db_url)
    monkeypatch.setattr(tokens, "HFTokenCounter", tokens.WhitespaceCounter)
    monkeypatch.setattr(embed, "LocalEmbedder", FakeEmbedder)
    assert cli.main(["db", "up"]) == 0
    code = cli.main(["ingest", "--versions", "16", "--raw-dir", FIXTURES])
    out = capsys.readouterr().out

    assert code == 0
    assert "pages: 4 new, 0 changed, 0 unchanged, 0 removed" in out
    assert "storage: pages" in out


def test_ingest_without_the_embed_group_explains_how_to_install(monkeypatch, capsys):
    def missing():
        raise ModuleNotFoundError("No module named 'torch'", name="torch")

    monkeypatch.setattr(tokens, "HFTokenCounter", tokens.WhitespaceCounter)
    monkeypatch.setattr(embed, "LocalEmbedder", missing)
    assert cli.main(["ingest", "--raw-dir", FIXTURES]) == 1
    assert "embed` dependency group" in capsys.readouterr().err


def test_golden_check_offline_passes_on_the_committed_set(capsys):
    golden_file = str(Path(__file__).parent.parent / "eval" / "golden.yaml")
    assert cli.main(["golden", "check", "--offline", "--file", golden_file]) == 0
    assert "OK: rules only" in capsys.readouterr().out


def test_golden_check_reports_problems(tmp_path, capsys):
    bad = tmp_path / "golden.yaml"
    bad.write_text(
        "format: 1\nquestions:\n  - id: q001\n    question: x?\n    versions: [16]\n"
        "    type: single\n    style: keyword\n    area: sql\n    source: human\n"
        "    status: reviewed\n    answer: y\n"
    )
    assert cli.main(["golden", "check", "--offline", "--file", str(bad)]) == 1
    assert "q001: a single question needs evidence" in capsys.readouterr().out
