"""Chunker tests. A whitespace "tokenizer" (one token per word) keeps the arithmetic obvious."""

from __future__ import annotations

from pathlib import Path

import pytest

from docs_qa import parse, tokens
from docs_qa.chunk import Chunker, balanced_groups, content_hash
from docs_qa.parse import Page, Section

FIXTURES = Path(__file__).parent / "fixtures"
COUNTER = tokens.WhitespaceCounter()


def words(n: int, word: str = "w") -> str:
    return " ".join([word] * n)


def make_page(blocks: tuple[str, ...], version: str = "16", key: str = "SQL-WITH", trail=None):
    trail = trail or ("SELECT", "WITH Clause")
    url = f"https://www.postgresql.org/docs/{version}/sql-select.html#{key}"
    section = Section(key, key, trail[-1], trail, url, blocks)
    return Page(version, "sql-select.html", url, "SELECT", None, "refentry", (section,))


def chunks_of(blocks, max_tokens=100, **kwargs):
    chunker = Chunker(COUNTER, max_tokens)
    return chunker.chunk_page(make_page(tuple(blocks), **kwargs)), chunker


def body(chunk) -> str:
    return chunk.text.split("\n\n", 1)[1]


# --- packing ------------------------------------------------------------------------------


def test_small_section_is_one_chunk_with_location_line():
    [chunk], _ = chunks_of(["First paragraph.", "Second paragraph."])
    assert chunk.text == "SELECT › WITH Clause\n\nFirst paragraph.\n\nSecond paragraph."
    assert chunk.id == "16/sql-select.html#SQL-WITH/0"
    assert chunk.token_count == COUNTER.count(chunk.text)


def test_blocks_are_never_split_when_they_fit():
    blocks = [words(30, f"b{i}") for i in range(10)]  # 300 tokens of 30-token blocks
    chunks, _ = chunks_of(blocks)
    rebuilt = [b for c in chunks for b in body(c).split("\n\n")]
    assert rebuilt == blocks


def test_no_chunk_exceeds_the_limit_location_line_included():
    blocks = [words(n) for n in (5, 60, 33, 97, 12, 40, 88, 1, 70)]
    chunks, _ = chunks_of(blocks, trail=("A long", "location line", "here"))
    assert all(c.token_count <= 100 for c in chunks)


def test_sizes_are_balanced_not_greedy():
    # 9 blocks of 20 tokens = 180; 96 left after the 4-token location line, so at most
    # 4 blocks per chunk and 3 chunks are needed. Greedy gives 80 + 80 + 20; balanced 60 x 3.
    chunks, _ = chunks_of([words(20) for _ in range(9)])
    assert [len(body(c).split()) for c in chunks] == [60, 60, 60]


def test_balanced_groups_aims_for_equal_sizes():
    pieces = [(f"p{i}", 100) for i in range(9)]
    groups = balanced_groups(pieces, budget=400)
    assert [sum(t for _, t in g) for g in groups] == [300, 300, 300]


def test_balanced_groups_never_exceeds_budget():
    pieces = [(str(i), t) for i, t in enumerate([90, 15, 60, 60, 99, 3, 40, 70, 25])]
    for budget in (99, 120, 150, 400):
        assert all(sum(t for _, t in g) <= budget for g in balanced_groups(pieces, budget))


def test_ordinals_and_ids_count_up_within_a_section():
    chunks, _ = chunks_of([words(60) for _ in range(3)])
    assert [c.id for c in chunks] == [f"16/sql-select.html#SQL-WITH/{i}" for i in range(3)]
    assert [c.ordinal for c in chunks] == [0, 1, 2]


def test_sections_are_never_merged():
    page = parse.parse_file(FIXTURES, "16", "runtime-config-autovacuum.html")
    chunks = Chunker(COUNTER, 400).chunk_page(page)
    assert [c.section_key for c in chunks] == [s.key for s in page.sections]
    naptime = next(c for c in chunks if c.section_key == "GUC-AUTOVACUUM-NAPTIME")
    assert naptime.text.startswith("Automatic Vacuuming › autovacuum_naptime (integer)\n\n")
    assert naptime.url.endswith("#GUC-AUTOVACUUM-NAPTIME")


# --- oversized blocks ---------------------------------------------------------------------


def test_oversized_code_is_split_between_lines_with_fences_on_each_piece():
    lines = [f"SELECT {i}, a, b, c FROM t;" for i in range(40)]  # 40 lines x 6 tokens
    code = "```\n" + "\n".join(lines) + "\n```"
    chunks, chunker = chunks_of([code])
    assert len(chunks) > 1
    assert chunker.stats.split_code == 1 and chunker.stats.token_cuts == 0
    for c in chunks:
        assert body(c).startswith("```\n") and body(c).endswith("\n```")
        assert c.token_count <= 100
    rebuilt = [line for c in chunks for line in body(c)[4:-4].split("\n")]
    assert rebuilt == lines


def test_oversized_prose_is_split_between_sentences():
    sentences = [f"Sentence number {i} has exactly eight words here." for i in range(30)]
    chunks, chunker = chunks_of([" ".join(sentences)])
    assert chunker.stats.split_prose == 1 and chunker.stats.token_cuts == 0
    assert all(body(c).endswith("here.") for c in chunks)
    assert " ".join(body(c) for c in chunks) == " ".join(sentences)


def test_a_sentence_bigger_than_the_limit_is_cut_by_tokens():
    chunks, chunker = chunks_of([words(250)])
    assert chunker.stats.token_cuts == 1
    assert all(c.token_count <= 100 for c in chunks)
    assert sum(len(body(c).split()) for c in chunks) == 250


def test_a_huge_heading_is_shortened():
    chunks, _ = chunks_of(["Text."], trail=(words(60, "h"),))
    header = chunks[0].text.split("\n\n")[0]
    assert header.endswith(" …") and COUNTER.count(header) <= 26


# --- hashes and versions ------------------------------------------------------------------


def test_same_section_in_two_versions_has_the_same_hash_but_different_ids():
    blocks = ("Unchanged text.",)
    [a], _ = chunks_of(blocks, version="16")
    [b], _ = chunks_of(blocks, version="17")
    assert a.content_hash == b.content_hash == content_hash(a.text)
    assert a.id != b.id
    assert "16" not in a.text  # the version is metadata, not part of the embedded text


def test_changed_text_changes_the_hash():
    [a], _ = chunks_of(("Old wording.",))
    [b], _ = chunks_of(("New wording.",))
    assert a.content_hash != b.content_hash


def test_chunking_is_deterministic():
    page = parse.parse_file(FIXTURES, "16", "sql-select.html")
    first = Chunker(COUNTER, 120).chunk_page(page)
    second = Chunker(COUNTER, 120).chunk_page(parse.parse_file(FIXTURES, "16", "sql-select.html"))
    assert first == second


def test_max_tokens_has_a_floor():
    with pytest.raises(ValueError):
        Chunker(COUNTER, 10)


# --- the real tokenizer (downloads ~700 KB once; skipped when that isn't possible) ---------


@pytest.fixture(scope="module")
def real_counter():
    try:
        return tokens.HFTokenCounter()
    except Exception as e:
        pytest.skip(f"real tokenizer unavailable: {e}")


def test_real_tokenizer_counts_past_512(real_counter):
    text = "autovacuum " * 1000
    assert real_counter.count(text) >= 1000  # not truncated at the model's 512


def test_real_tokenizer_cut_pieces_fit(real_counter):
    text = "The quick brown fox jumps over the lazy dog. " * 100
    pieces = real_counter.cut(text, 50)
    assert len(pieces) > 1
    assert all(real_counter.count(p) <= 52 for p in pieces)  # re-tokenizing may shift by one


def test_real_tokenizer_chunks_fit(real_counter):
    page = parse.parse_file(FIXTURES, "16", "sql-select.html")
    chunks = Chunker(real_counter, 400).chunk_page(page)
    assert chunks and all(c.token_count <= 400 for c in chunks)
