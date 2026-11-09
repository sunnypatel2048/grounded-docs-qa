import pytest
from conftest import require_pgvector

from docs_qa import cli


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
