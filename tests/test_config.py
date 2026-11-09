import pytest

from docs_qa import cli, config


@pytest.fixture
def no_env(monkeypatch, tmp_path):
    """No database variables in the shell, and a working directory with no .env above it."""
    for name in ("DATABASE_URL", "TEST_DATABASE_URL"):
        monkeypatch.delenv(name, raising=False)  # also restores the originals afterwards
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_missing_database_url_explains_the_fix(no_env):
    with pytest.raises(config.ConfigError, match="Copy .env.example to .env"):
        config.database_url()


def test_database_url_is_read_from_dot_env(no_env):
    (no_env / ".env").write_text("DATABASE_URL=postgresql://u:p@localhost:5433/docsqa\n")
    assert config.database_url() == "postgresql://u:p@localhost:5433/docsqa"


def test_shell_variable_wins_over_dot_env(no_env, monkeypatch):
    (no_env / ".env").write_text("DATABASE_URL=postgresql://from-file/docsqa\n")
    monkeypatch.setenv("DATABASE_URL", "postgresql://from-shell/docsqa")
    assert config.database_url() == "postgresql://from-shell/docsqa"


def test_test_database_url_is_none_when_unset(no_env):
    assert config.test_database_url() is None


@pytest.mark.parametrize(
    ("url", "name"),
    [
        ("postgresql://docsqa:docsqa@localhost:5433/docsqa_test", "docsqa_test"),
        ("postgresql://docsqa@/docsqa_test?host=/var/tmp&port=55432", "docsqa_test"),
        ("postgresql://localhost/docsqa", "docsqa"),
    ],
)
def test_database_name(url, name):
    assert config.database_name(url) == name


def test_test_database_must_end_in_test():
    with pytest.raises(config.ConfigError, match="must end in '_test'"):
        config.check_test_database("postgresql://localhost/docsqa", None)


def test_test_database_must_differ_from_the_real_one():
    # Both end in _test here, so only the "same database" rule can catch it.
    with pytest.raises(config.ConfigError, match="both point at"):
        config.check_test_database(
            "postgresql://localhost:5433/docsqa_test", "postgresql://other:5433/docsqa_test"
        )


def test_a_proper_test_database_passes():
    config.check_test_database(
        "postgresql://localhost:5433/docsqa_test", "postgresql://localhost:5433/docsqa"
    )


def test_cli_reports_missing_config_in_one_line(no_env, capsys):
    assert cli.main(["db", "up"]) == 1
    assert "error: DATABASE_URL isn't set" in capsys.readouterr().err
