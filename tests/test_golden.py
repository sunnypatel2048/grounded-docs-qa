"""Golden set tests: the format rules, quote matching, and checking evidence against stored docs."""

from __future__ import annotations

import copy
import shutil
from pathlib import Path

import pytest
from conftest import require_pgvector

from docs_qa import db, golden, ingest, tokens
from docs_qa.chunk import Chunker
from docs_qa.embed import HashEmbedder
from docs_qa.golden import Evidence, GoldenError, chunk_has_evidence, normalize, parse_data

ROOT = Path(__file__).parent.parent
FIXTURES = Path(__file__).parent / "fixtures" / "16"

NAPTIME = "16/runtime-config-autovacuum.html#GUC-AUTOVACUUM-NAPTIME"
VALID = {
    "id": "q001",
    "question": "What is the default value of autovacuum_naptime in PostgreSQL 16?",
    "versions": ["16"],
    "type": "single",
    "style": "keyword",
    "area": "settings",
    "source": "human",
    "status": "reviewed",
    "evidence": [{"ref": NAPTIME, "quote": "The default is one minute"}],
    "answer": "One minute.",
}


def problems_for(**changes) -> list[str]:
    item = copy.deepcopy(VALID)
    for key, value in changes.items():
        if value is None:
            item.pop(key, None)
        else:
            item[key] = value
    return parse_data({"format": 1, "questions": [item]}).problems


# --- the committed file ---------------------------------------------------------------------


def test_the_committed_golden_set_follows_the_rules():
    gold = golden.load(ROOT / "eval" / "golden.yaml")
    assert gold.problems == []
    assert gold.questions


# --- format rules ---------------------------------------------------------------------------


def test_a_valid_question_has_no_problems():
    gold = parse_data({"format": 1, "questions": [VALID]})
    assert gold.problems == []
    [q] = gold.questions
    assert q.evidence == (Evidence(NAPTIME, "The default is one minute"),)
    assert q.evidence[0].parts == ("16", "runtime-config-autovacuum.html", "GUC-AUTOVACUUM-NAPTIME")


def test_versions_written_as_numbers_are_accepted():
    [q] = parse_data({"format": 1, "questions": [{**VALID, "versions": [16]}]}).questions
    assert q.versions == ("16",)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"answer": None}, "missing answer"),
        ({"id": "Q-1"}, "id must look like q001"),
        ({"type": "fact"}, "type must be one of"),
        ({"style": "fuzzy"}, "style must be one of"),
        ({"area": "misc"}, "area must be one of"),
        ({"status": "done"}, "status must be one of"),
        ({"versions": ["sixteen"]}, "major versions"),
        ({"surprise": 1}, "unknown fields surprise"),
        ({"evidence": []}, "needs evidence"),
        ({"evidence": [{"ref": "runtime.html", "quote": "abc"}]}, "must look like 16/page.html"),
        ({"evidence": [{"ref": NAPTIME, "quote": "ab"}]}, "short phrase"),
        ({"evidence": [{"ref": NAPTIME}]}, "needs a ref and a quote"),
        ({"versions": ["17"]}, "from a version the question isn't about"),
        ({"versions": ["16", "17"]}, "exactly one version"),
        ({"type": "multi_hop"}, "at least 2 sections"),
        ({"type": "version_diff", "versions": ["16", "17"]}, "at least 2 versions"),
        ({"type": "unanswerable"}, "has no evidence"),
    ],
)
def test_rule_violations_are_reported(changes, message):
    problems = problems_for(**changes)
    assert any(message in p for p in problems), problems
    assert all(p.startswith(("q001:", "Q-1:")) for p in problems)


def test_duplicate_ids_are_reported():
    gold = parse_data({"format": 1, "questions": [VALID, VALID]})
    assert gold.problems == ["q001: duplicate id"]


def test_an_unanswerable_question_needs_no_evidence():
    assert problems_for(type="unanswerable", evidence=None) == []


@pytest.mark.parametrize(
    "data", [None, [], {"questions": []}, {"format": 2, "questions": []}, {"format": 1}]
)
def test_a_file_of_the_wrong_shape_is_refused(data):
    with pytest.raises(GoldenError):
        parse_data(data)


def test_invalid_yaml_is_refused(tmp_path):
    path = tmp_path / "golden.yaml"
    path.write_text("format: 1\nquestions: [unclosed\n")
    with pytest.raises(GoldenError, match="isn't valid YAML"):
        golden.load(path)


def test_split_is_fixed_by_id_and_about_30_percent_test():
    def question(i):
        return parse_data({"format": 1, "questions": [{**VALID, "id": f"q{i:04d}"}]}).questions[0]

    splits = [question(i).split for i in range(1000)]
    assert 250 <= splits.count("test") <= 350
    assert question(7).split == question(7).split


def test_mix_report_compares_against_targets():
    lines = golden.mix_report(parse_data({"format": 1, "questions": [VALID]}))
    assert lines[0] == "1 questions, 1 reviewed, 0 drafts"
    assert any("single" in line and "100%" in line and "target 60%" in line for line in lines)


# --- matching quotes in retrieved chunks ----------------------------------------------------


def test_normalize_ignores_case_curly_quotes_and_whitespace():
    assert normalize("  The “AS-IS”\n  Basis\xa0") == 'the "as-is" basis'


def test_chunk_has_evidence_needs_the_right_section_and_the_quote():
    e = Evidence(NAPTIME, "the default is ONE minute")
    text = "Automatic Vacuuming › autovacuum_naptime\n\n... The default is one   minute (1min)."
    page = "runtime-config-autovacuum.html"
    assert chunk_has_evidence("16", page, "GUC-AUTOVACUUM-NAPTIME", text, e)
    assert not chunk_has_evidence("17", page, "GUC-AUTOVACUUM-NAPTIME", text, e)
    assert not chunk_has_evidence("16", page, "GUC-AUTOVACUUM", text, e)
    assert not chunk_has_evidence("16", page, "GUC-AUTOVACUUM-NAPTIME", "Unrelated.", e)


# --- evidence against the stored docs -------------------------------------------------------


@pytest.fixture
def stored(db_conn, tmp_path):
    require_pgvector(db_conn)
    db.migrate(db_conn)
    shutil.copytree(FIXTURES, tmp_path / "16")
    ingest.ingest(
        db_conn,
        raw_dir=tmp_path,
        versions=["16"],
        chunker=Chunker(tokens.WhitespaceCounter(), 400),
        embedder=HashEmbedder(),
        log=lambda _m: None,
    )
    return db_conn


def check(conn, evidence):
    item = {**VALID, "evidence": evidence}
    return golden.evidence_problems(conn, parse_data({"format": 1, "questions": [item]}))


@pytest.mark.db
def test_evidence_that_exists_passes(stored):
    assert check(stored, [{"ref": NAPTIME, "quote": "the default is ONE minute (1min)"}]) == []


@pytest.mark.db
def test_a_quote_not_in_the_section_is_reported(stored):
    [problem] = check(stored, [{"ref": NAPTIME, "quote": "The default is two minutes"}])
    assert problem.startswith("q001: quote not found in " + NAPTIME)


@pytest.mark.db
def test_a_quote_from_another_section_is_reported(stored):
    # The quote exists on the page, but in autovacuum_max_workers, not autovacuum_naptime.
    [problem] = check(stored, [{"ref": NAPTIME, "quote": "The default is three"}])
    assert "quote not found" in problem


@pytest.mark.db
def test_the_location_line_does_not_count_as_section_text(stored):
    [problem] = check(stored, [{"ref": NAPTIME, "quote": "Automatic Vacuuming › autovacuum"}])
    assert "quote not found" in problem


@pytest.mark.db
def test_a_section_that_does_not_exist_is_reported(stored):
    ref = "16/runtime-config-autovacuum.html#GUC-NO-SUCH-SETTING"
    assert check(stored, [{"ref": ref, "quote": "anything"}]) == [f"q001: no stored section {ref}"]
