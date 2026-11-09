import shutil

import pytest
from conftest import require_pgvector

from docs_qa import db


def write(directory, name, sql="SELECT 1;"):
    path = directory / name
    path.write_text(sql)
    return path


# --- discovery (no database needed) -------------------------------------------------------


def test_discover_orders_by_number(tmp_path):
    write(tmp_path, "010_later.sql")
    write(tmp_path, "002_second.sql")
    write(tmp_path, "001_first.sql")
    assert [m.version for m in db.discover(tmp_path)] == [1, 2, 10]


def test_discover_rejects_badly_named_files(tmp_path):
    write(tmp_path, "1_init.sql")
    with pytest.raises(db.MigrationError, match="NNN_name.sql"):
        db.discover(tmp_path)


def test_discover_rejects_duplicate_numbers(tmp_path):
    write(tmp_path, "001_a.sql")
    write(tmp_path, "001_b.sql")
    with pytest.raises(db.MigrationError, match="duplicate"):
        db.discover(tmp_path)


def test_checksum_changes_when_the_file_changes(tmp_path):
    path = write(tmp_path, "001_a.sql", "SELECT 1;")
    before = db.discover(tmp_path)[0].checksum
    path.write_text("SELECT 2;")
    assert db.discover(tmp_path)[0].checksum != before


def test_the_real_migrations_are_well_formed():
    migrations = db.discover()
    assert migrations[0].path.name == "001_init.sql"


# --- applying (needs Postgres) -------------------------------------------------------------


def two_tables(tmp_path):
    write(tmp_path, "001_one.sql", "CREATE TABLE one (id int);")
    write(tmp_path, "002_two.sql", "CREATE TABLE two (id int);")
    return db.discover(tmp_path)


@pytest.mark.db
def test_migrate_applies_everything_once(db_conn, tmp_path):
    migrations = two_tables(tmp_path)
    assert [m.version for m in db.migrate(db_conn, migrations)] == [1, 2]
    assert db.migrate(db_conn, migrations) == []  # second run: nothing to do
    assert sorted(db.applied(db_conn)) == [1, 2]


@pytest.mark.db
def test_new_migrations_are_picked_up_later(db_conn, tmp_path):
    db.migrate(db_conn, two_tables(tmp_path))
    write(tmp_path, "003_three.sql", "CREATE TABLE three (id int);")
    assert [m.version for m in db.migrate(db_conn, db.discover(tmp_path))] == [3]


@pytest.mark.db
def test_a_failing_migration_leaves_the_database_unchanged(db_conn, tmp_path):
    write(tmp_path, "001_ok.sql", "CREATE TABLE ok (id int);")
    write(tmp_path, "002_broken.sql", "CREATE TABLE broken (id int; -- syntax error")
    import psycopg

    with pytest.raises(psycopg.errors.SyntaxError):
        db.migrate(db_conn, db.discover(tmp_path))
    db_conn.rollback()
    assert db.applied(db_conn) == {}
    assert db_conn.execute("SELECT to_regclass('ok')").fetchone()[0] is None


@pytest.mark.db
def test_editing_an_applied_migration_is_an_error(db_conn, tmp_path):
    migrations = two_tables(tmp_path)
    db.migrate(db_conn, migrations)
    (tmp_path / "001_one.sql").write_text("CREATE TABLE one (id bigint);")
    with pytest.raises(db.MigrationError, match="edited after it was applied"):
        db.migrate(db_conn, db.discover(tmp_path))


@pytest.mark.db
def test_a_deleted_migration_file_is_an_error(db_conn, tmp_path):
    db.migrate(db_conn, two_tables(tmp_path))
    (tmp_path / "002_two.sql").unlink()
    with pytest.raises(db.MigrationError, match="file is gone"):
        db.pending(db_conn, db.discover(tmp_path))


@pytest.mark.db
def test_the_real_schema_applies_cleanly(db_conn, tmp_path):
    # Needs the vector extension, so it runs against the pgvector image (docker compose, CI).
    require_pgvector(db_conn)
    shutil.copytree(db.MIGRATIONS_DIR, tmp_path / "sql")
    applied = db.migrate(db_conn, db.discover(tmp_path / "sql"))
    assert applied[0].name == "init"
    tables = {
        row[0]
        for row in db_conn.execute(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
        ).fetchall()
    }
    assert {"pages", "chunks", "embedding_cache", "schema_migrations"} <= tables
    assert db.migrate(db_conn, db.discover(tmp_path / "sql")) == []
