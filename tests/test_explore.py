"""Tests for sampling sections and finding changes between versions."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from conftest import require_pgvector

from docs_qa import cli, db, explore, ingest, tokens
from docs_qa.chunk import Chunker
from docs_qa.embed import HashEmbedder
from docs_qa.explore import SectionText

FIXTURES = Path(__file__).parent / "fixtures" / "16"
LONG = " ".join(["word"] * 50)


def section(page, key, text=LONG, version="16", tokens=None):
    url = f"https://www.postgresql.org/docs/{version}/{page}"
    return SectionText(version, page, key, ("Trail",), url, text, tokens or len(text.split()))


# --- areas and sampling ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("page", "area"),
    [
        ("runtime-config-autovacuum.html", "settings"),
        ("sql-select.html", "sql"),
        ("functions-string.html", "functions"),
        ("catalog-pg-class.html", "catalogs"),
        ("app-psql.html", "clients"),
        ("hstore.html", "extensions"),
        ("contrib-dblink-connect.html", "extensions"),
        ("release-17-1.html", "release_notes"),
        ("mvcc-intro.html", "internals"),
        ("backup-dump.html", "admin"),
        ("bug-reporting.html", "other"),
    ],
)
def test_area_for(page, area):
    assert explore.area_for(page) == area


def test_sample_takes_areas_in_turn():
    sections = [section(f"{prefix}-{i}.html", f"K{i}") for prefix in ("sql", "functions", "app")
                for i in range(5)]  # fmt: skip
    picked = explore.sample(sections, 3, seed=1)
    assert sorted(explore.area_for(s.page) for s in picked) == ["clients", "functions", "sql"]


def test_sample_skips_short_and_already_used_sections():
    short = section("sql-a.html", "SHORT", "too short", tokens=5)
    used = section("sql-b.html", "USED")
    fresh = section("sql-c.html", "FRESH")
    picked = explore.sample([short, used, fresh], 10, seed=0, exclude={used.ref})
    assert picked == [fresh]


def test_sample_is_repeatable_for_a_seed():
    sections = [section(f"sql-{i}.html", f"K{i}") for i in range(40)]
    assert explore.sample(sections, 5, seed=3) == explore.sample(list(reversed(sections)), 5, 3)
    assert explore.sample(sections, 5, seed=3) != explore.sample(sections, 5, seed=4)


# --- changes between versions ---------------------------------------------------------------


def test_identical_sections_are_not_changes():
    found, new = explore.changes([section("a.html", "K")], [section("a.html", "K", version="17")])
    assert found == [] and new == []


def test_renumbered_cross_references_are_not_changes():
    old = section("a.html", "K", LONG + " See Section 25.1.6 and Table 9.9.")
    cur = section("a.html", "K", LONG + " See Section 26.1.6 and Table 9.10.", version="17")
    assert explore.changes([old], [cur], min_words=1)[0] == []


def test_a_real_change_reports_original_sentences():
    old = section("a.html", "K", LONG + ". The default is three. See Section 25.1.6.")
    cur = section(
        "a.html", "K", LONG + ". The default is 3 and it can be reloaded. See Section 26.1.6.",
        version="17",
    )  # fmt: skip
    [change] = explore.changes([old], [cur], min_words=1)[0]
    assert change.removed == ("The default is three.",)
    assert change.added == ("The default is 3 and it can be reloaded.",)
    assert change.changed_words == 4 + 9


def test_small_changes_are_ignored_by_default():
    old = section("a.html", "K", LONG + ". Default three.")
    cur = section("a.html", "K", LONG + ". Default 3.", version="17")
    assert explore.changes([old], [cur])[0] == []  # 4 words changed; 8 is the minimum
    assert len(explore.changes([old], [cur], min_words=4)[0]) == 1


def test_a_section_on_a_renamed_page_is_matched_by_key():
    old = section("runtime-config-autovacuum.html", "GUC-AUTOVACUUM-MAX-WORKERS", LONG + ". Old.")
    cur = section(
        "runtime-config-vacuum.html", "GUC-AUTOVACUUM-MAX-WORKERS",
        LONG + ". New text that is long enough to count as a change.", version="18",
    )  # fmt: skip
    found, new = explore.changes([old], [cur])
    assert [c.after.ref for c in found] == [
        "18/runtime-config-vacuum.html#GUC-AUTOVACUUM-MAX-WORKERS"
    ]
    assert new == []


def test_new_sections_are_listed_but_not_short_ones():
    cur = [
        section("a.html", "NEW", version="17"),
        section("a.html", "TINY", "short", version="17", tokens=3),
    ]
    found, new = explore.changes([], cur)
    assert [s.key for s in new] == ["NEW"]


# --- against the stored docs ----------------------------------------------------------------


@pytest.fixture
def stored(db_conn, tmp_path):
    """Fixture pages stored as version 16, and as version 17 with one setting reworded."""
    require_pgvector(db_conn)
    db.migrate(db_conn)
    shutil.copytree(FIXTURES, tmp_path / "16")
    shutil.copytree(FIXTURES, tmp_path / "17")
    page = tmp_path / "17" / "runtime-config-autovacuum.html"
    html = page.read_text(encoding="utf-8")
    page.write_text(
        html.replace(
            "The default is one minute",
            "The default is two minutes, and it can now be set per database with ALTER DATABASE",
        ),
        encoding="utf-8",
    )
    ingest.ingest(
        db_conn,
        raw_dir=tmp_path,
        versions=["16", "17"],
        chunker=Chunker(tokens.WhitespaceCounter(), 400),
        embedder=HashEmbedder(),
        log=lambda _m: None,
    )
    return db_conn


@pytest.mark.db
def test_load_sections_joins_chunks_without_location_lines(stored):
    sections = {s.ref: s for s in explore.load_sections(stored, ["16"])}
    naptime = sections["16/runtime-config-autovacuum.html#GUC-AUTOVACUUM-NAPTIME"]
    assert naptime.text.startswith("Specifies the minimum delay")
    assert naptime.trail == ("Automatic Vacuuming", "autovacuum_naptime (integer)")
    assert all(s.version == "16" for s in sections.values())


@pytest.mark.db
def test_changes_finds_the_reworded_setting(stored):
    sections = explore.load_sections(stored, ["16", "17"])
    found, new = explore.changes(
        [s for s in sections if s.version == "16"], [s for s in sections if s.version == "17"]
    )
    assert [c.after.key for c in found] == ["GUC-AUTOVACUUM-NAPTIME"]
    assert any("two minutes" in s for s in found[0].added)
    assert new == []


@pytest.mark.db
def test_cli_sample_and_changes(stored, test_db_url, monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("DATABASE_URL", test_db_url)
    no_golden = str(tmp_path / "none.yaml")
    assert cli.main(["golden", "sample", "-n", "3", "--versions", "16", "--file", no_golden]) == 0
    out = capsys.readouterr().out
    assert out.count("─── 16/") == 3 and "run with --seed 1" in out

    assert cli.main(["golden", "changes", "--from", "16", "--to", "17", "--file", no_golden]) == 0
    out = capsys.readouterr().out
    assert "1 sections changed from 16 to 17" in out
    assert "+ 17: " in out and "two minutes" in out


@pytest.mark.db
def test_cli_sample_skips_sections_already_in_the_golden_set(
    stored, test_db_url, monkeypatch, capsys, tmp_path
):
    monkeypatch.setenv("DATABASE_URL", test_db_url)
    used = "16/runtime-config-autovacuum.html#GUC-AUTOVACUUM-NAPTIME"
    golden_file = tmp_path / "golden.yaml"
    golden_file.write_text(
        "format: 1\nquestions:\n  - id: q001\n    question: x?\n    versions: [16]\n"
        "    type: single\n    style: keyword\n    area: settings\n    source: human\n"
        f'    status: reviewed\n    answer: y\n    evidence:\n      - ref: "{used}"\n'
        "        quote: The default is one minute\n"
    )
    args = ["golden", "sample", "-n", "50", "--versions", "16", "--file", str(golden_file)]
    assert cli.main(args) == 0
    assert used not in capsys.readouterr().out
