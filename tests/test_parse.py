"""Parser tests: trimmed real pages in tests/fixtures/16/, plus small hand-written pages."""

from __future__ import annotations

from pathlib import Path

import pytest

from docs_qa import parse
from docs_qa.parse import ParseError, parse_page

FIXTURES = Path(__file__).parent / "fixtures"
DOCS = "https://www.postgresql.org/docs/16/"


def fixture(name: str) -> parse.Page:
    return parse.parse_file(FIXTURES, "16", name)


def by_key(page: parse.Page) -> dict[str, parse.Section]:
    return {s.key: s for s in page.sections}


def doc(content: str, title: str = "Test Page", home: str = "PostgreSQL 16.15 Documentation"):
    """A minimal page in the shape postgresql.org serves."""
    return f"""<html><head><title>PostgreSQL: Documentation: 16: {title}</title></head><body>
    <div id="pgContentWrap">Supported Versions: Current (18) / 17 / 16</div>
    <div id="docContent">
      <div class="navheader"><a accesskey="h" href="index.html" title="{home}">Home</a></div>
      {content}
      <div class="navfooter"><a href="next.html">Next</a></div>
    </div></body></html>"""


def sect(id_: str, heading: str, body: str, cls: str = "sect1", h: str = "h2") -> str:
    id_attr = f' id="{id_}"' if id_ else ""
    return (
        f'<div class="{cls}"{id_attr}><div class="titlepage"><{h} class="title">{heading}'
        f' <a href="#{id_}" class="id_link">#</a></{h}></div>{body}</div>'
    )


def all_text(page: parse.Page) -> str:
    return "\n".join(b for s in page.sections for b in s.blocks)


# --- real pages ---------------------------------------------------------------------------


def test_page_metadata():
    page = fixture("bug-reporting.html")
    assert page.title == "5. Bug Reporting Guidelines"
    assert page.release == "16.15"
    assert page.kind == "sect1"
    assert page.url == DOCS + "bug-reporting.html"


def test_site_chrome_and_navigation_are_dropped():
    text = all_text(fixture("bug-reporting.html"))
    assert "Supported Versions" not in text
    assert "Prev" not in text and "Submit correction" not in text
    assert "5.1. Identifying Bugs" not in text  # the table of contents
    assert " #" not in text  # permalink icons


def test_sections_follow_the_section_divs():
    page = fixture("bug-reporting.html")
    assert [s.key for s in page.sections] == [
        "BUG-REPORTING",
        "BUG-REPORTING-IDENTIFYING-BUGS",
        "BUG-REPORTING-WHAT-TO-REPORT",
    ]
    section = page.sections[1]
    assert section.heading == "5.1. Identifying Bugs"
    assert section.trail == ("Bug Reporting Guidelines", "Identifying Bugs")
    assert section.url == DOCS + "bug-reporting.html#BUG-REPORTING-IDENTIFYING-BUGS"


def test_notes_stay_inside_their_section():
    page = fixture("bug-reporting.html")
    assert not any(s.heading == "Note" for s in page.sections)
    blocks = by_key(page)["BUG-REPORTING-WHAT-TO-REPORT"].blocks
    assert any("Note: If you are reporting an error message" in b for b in blocks)


def test_list_items_are_separate_blocks():
    blocks = by_key(fixture("bug-reporting.html"))["BUG-REPORTING-IDENTIFYING-BUGS"].blocks
    assert "- A program produces the wrong output for any given input." in blocks


def test_reference_page_sections_get_stable_keys():
    sections = by_key(fixture("sql-select.html"))
    assert list(sections) == [
        "SQL-SELECT",
        "SQL-SELECT/synopsis",
        "SQL-SELECT/description",
        "SQL-WITH",
        "SQL-FROM",
        "SQL-SELECT/examples",
    ]
    description = sections["SQL-SELECT/description"]
    assert description.anchor == "id-1.9.3.172.7"  # the real id, still used for the link
    assert description.url.endswith("sql-select.html#id-1.9.3.172.7")
    assert sections["SQL-WITH"].trail == ("SELECT", "Parameters", "WITH Clause")


def test_reference_page_title_and_purpose():
    root = by_key(fixture("sql-select.html"))["SQL-SELECT"]
    assert root.heading == "SELECT"
    assert root.blocks == ("SELECT, TABLE, WITH — retrieve rows from a table or view",)


def test_code_keeps_its_line_breaks_and_indentation():
    synopsis = by_key(fixture("sql-select.html"))["SQL-SELECT/synopsis"].blocks[0]
    assert synopsis.startswith("```\n[ WITH [ RECURSIVE ] with_query [, ...] ]\nSELECT")
    assert "\n    [ FROM from_item [, ...] ]\n" in synopsis
    assert synopsis.endswith("\n```")


def test_each_setting_is_its_own_section():
    sections = by_key(fixture("runtime-config-autovacuum.html"))
    naptime = sections["GUC-AUTOVACUUM-NAPTIME"]
    assert naptime.heading == "autovacuum_naptime (integer)"
    assert naptime.trail == ("Automatic Vacuuming", "autovacuum_naptime (integer)")
    assert naptime.url == DOCS + "runtime-config-autovacuum.html#GUC-AUTOVACUUM-NAPTIME"
    assert naptime.blocks[0].startswith("Specifies the minimum delay between autovacuum runs")
    # the settings' text isn't repeated in the parent section
    assert "minimum delay" not in "\n".join(sections["RUNTIME-CONFIG-AUTOVACUUM"].blocks)


def test_function_table_rows_stay_whole():
    blocks = fixture("functions-string.html").sections[0].blocks
    assert "Table 9.9. SQL String Functions and Operators" in blocks
    assert (
        "text || text → text\nConcatenates the two strings.\n'Post' || 'greSQL' → PostgreSQL"
        in blocks
    )


# --- hand-written pages -------------------------------------------------------------------


def test_generated_ids_are_keyed_by_parent_and_heading():
    html = doc(
        sect(
            "APP-PGDUMP",
            "pg_dump",
            sect("id-1.9.4.13.7", "Environment", "<p>PGHOST.</p>", "refsect1")
            + sect("id-1.9.4.13.8", "Notes", "<p>One.</p>", "refsect1")
            + sect("id-1.9.4.13.9", "Notes", "<p>Two.</p>", "refsect1"),
            "refentry",
        )
    )
    keys = [s.key for s in parse_page(html, "16", "app-pgdump.html").sections]
    assert keys == ["APP-PGDUMP/environment", "APP-PGDUMP/notes", "APP-PGDUMP/notes-2"]


def test_keys_ignore_section_numbers_so_they_survive_renumbering():
    html = doc(sect("id-1.2.3", "7.8. WITH Queries", "<p>Text.</p>"))
    section = parse_page(html, "16", "queries-with.html").sections[0]
    assert section.key == "with-queries"
    assert section.heading == "7.8. WITH Queries"
    assert section.trail == ("WITH Queries",)


def test_table_of_contents_page_has_no_sections():
    toc = '<div class="toc"><dl class="toc"><dt><a href="a.html">1. A</a></dt></dl></div>'
    page = parse_page(doc(sect("TUTORIAL", "Part I. Tutorial", toc, "part")), "16", "t.html")
    assert page.sections == ()
    assert page.kind == "part"


def test_book_index_is_skipped():
    html = doc('<div class="index" id="BOOKINDEX"><dl><dt>ABORT, 1</dt></dl></div>')
    page = parse_page(html, "16", "bookindex.html")
    assert page.kind == "index" and page.sections == ()


def test_multi_column_rows_are_labelled_with_their_headers():
    table = """<div class="table"><p class="title"><strong>Table 8.1. Data Types</strong></p>
      <table><thead><tr><th>Name</th><th>Aliases</th><th>Description</th></tr></thead>
      <tbody><tr><td><code>bigint</code></td><td><code>int8</code></td><td>eight bytes</td></tr>
      <tr><td><code>box</code></td><td>&nbsp;</td><td>rectangular box</td></tr></tbody>
      </table></div>"""
    blocks = (
        parse_page(doc(sect("DATATYPE", "Data Types", table)), "16", "d.html").sections[0].blocks
    )
    assert blocks == (
        "Table 8.1. Data Types",
        "Name: bigint | Aliases: int8 | Description: eight bytes",
        "Name: box | Description: rectangular box",  # empty cell left out
    )


def test_definitions_without_ids_stay_in_the_section():
    dl = '<dl class="variablelist"><dt><code>-v</code></dt><dd><p>Be verbose.</p></dd></dl>'
    section = parse_page(doc(sect("APP-X", "x", dl)), "16", "x.html").sections[0]
    assert section.blocks == ("-v: Be verbose.",)


def test_whitespace_and_non_breaking_spaces_are_normalized():
    body = "<p>  Spread\n   over\xa0lines  </p><p> </p>"
    section = parse_page(doc(sect("X", "X", body)), "16", "x.html").sections[0]
    assert section.blocks == ("Spread over lines",)


def test_release_is_none_when_the_page_does_not_say():
    html = doc(sect("X", "X", "<p>Text.</p>"), home="Home")
    assert parse_page(html, "16", "x.html").release is None


def test_page_without_doc_content_is_an_error():
    with pytest.raises(ParseError, match="docContent"):
        parse_page("<html><body><p>Not a docs page</p></body></html>", "16", "x.html")


@pytest.mark.parametrize(
    ("value", "stable"),
    [("SQL-WITH", True), ("GUC-WORK-MEM", True), ("id-1.9.3.172.7", False), (None, False)],
)
def test_is_stable_id(value, stable):
    assert parse.is_stable_id(value) is stable
