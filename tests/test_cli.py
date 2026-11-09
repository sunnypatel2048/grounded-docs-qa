from pathlib import Path

import pytest
from conftest import require_pgvector

from docs_qa import cli, fetch, tokens


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
    assert "applied  001_init.sql" in out and "1 applied, 0 pending" in out


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
