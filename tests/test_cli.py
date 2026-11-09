import pytest
from conftest import require_pgvector

from docs_qa import cli, fetch


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
