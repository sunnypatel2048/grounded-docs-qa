"""Help for writing the golden set: random sections to write questions from, and what changed
between two versions of the docs (the source of version comparisons and version traps).

Sampling is spread across doc areas, so hand-written questions cover the docs instead of the
topics the writer already knows. Sections already used as evidence are skipped.
"""

from __future__ import annotations

import difflib
import random
import re
from collections import defaultdict
from dataclasses import dataclass

MIN_TOKENS = 40  # shorter sections rarely support a good question

# Doc area for a page, from its name. Only a suggestion for spreading samples across the docs;
# the question writer sets the real area.
_EXTENSIONS = frozenset(
    "adminpack amcheck auth-delay auto-explain basebackup-to-shell basic-archive bloom btree-gin "
    "btree-gist citext cube dblink dict-int dict-xsyn earthdistance file-fdw fuzzystrmatch hstore "
    "intagg intarray isn lo ltree oldsnapshot pageinspect passwordcheck pgbuffercache pgcrypto "
    "pgfreespacemap pglogicalinspect pgoverexplain pgprewarm pgrowlocks pgstatstatements "
    "pgstattuple pgsurgery pgtrgm pgvisibility pgwalinspect postgres-fdw seg sepgsql sslinfo "
    "tablefunc tcn test-decoding tsm-system-rows tsm-system-time unaccent uuid-ossp xml2".split()
)
_AREA_PREFIXES = (
    ("release_notes", ("release",)),
    ("extensions", ("contrib",)),
    ("settings", ("runtime-config",)),
    ("functions", ("functions",)),
    ("catalogs", ("catalog", "view-", "infoschema", "monitoring")),
    ("sql", ("sql-", "ddl", "dml", "queries", "datatype", "datetime", "typeconv", "indexes",
             "textsearch", "rowtypes", "arrays", "rangetypes", "tutorial", "collation", "rules",
             "xml", "glossary")),
    ("clients", ("app-", "libpq", "ecpg", "lo-", "pgbench", "oid2name", "vacuumlo")),
    ("admin", ("backup", "continuous-archiving", "high-availability", "warm-standby", "hot-standby",
               "logical-replication", "maintenance", "routine-", "upgrading", "server-", "auth",
               "client-authentication", "user-manag", "role", "manage-ag", "kernel-resources",
               "installation", "install-", "runtime", "creating-cluster", "disk", "wal-", "locale",
               "charset", "multibyte", "postgres-", "regress", "archive")),
    ("internals", ("storage", "mvcc", "planner", "indexam", "tableam", "xfunc", "xindex", "xaggr",
                   "xtypes", "xoper", "xplang", "spi", "plpgsql", "plpython", "pltcl", "plperl",
                   "trigger", "event-trigger", "protocol", "parallel", "geqo", "btree", "gin",
                   "gist", "spgist", "brin", "hash", "transaction", "explicit-locking",
                   "using-explain", "performance", "populate", "jit", "custom-", "fdw", "extend",
                   "logicaldecoding", "bgworker", "replication-origins", "tablesample")),
)  # fmt: skip
# Cross-references renumber whenever a chapter is added ("see Section 25.1.6"); they aren't
# real changes.
_XREF_NUMBER = re.compile(
    r"\b(Section|Table|Figure|Example|Chapter|Appendix|Part|Step)\s+[A-Z0-9]+(?:\.[0-9]+)*"
)
_SENTENCE = re.compile(r"(?<=[.!?:])\s+|\n+")


@dataclass(frozen=True)
class SectionText:
    version: str
    page: str
    key: str
    trail: tuple[str, ...]
    url: str
    text: str  # the section's chunks joined, location lines removed
    tokens: int

    @property
    def ref(self) -> str:
        return f"{self.version}/{self.page}#{self.key}"


@dataclass(frozen=True)
class Change:
    before: SectionText
    after: SectionText
    removed: tuple[str, ...]  # sentences only in `before`
    added: tuple[str, ...]  # sentences only in `after`
    changed_words: int


def area_for(page: str) -> str:
    if page.removesuffix(".html") in _EXTENSIONS:
        return "extensions"
    for area, prefixes in _AREA_PREFIXES:
        if page.startswith(prefixes):
            return area
    return "other"


def load_sections(conn, versions: list[str]) -> list[SectionText]:
    rows = conn.execute(
        "SELECT version, page_path, section_key, section_path, url, content, token_count "
        "FROM chunks WHERE version = ANY(%s) ORDER BY version, page_path, section_key, ordinal",
        (versions,),
    ).fetchall()
    conn.rollback()
    grouped: dict[tuple[str, str, str], list] = defaultdict(list)
    for row in rows:
        grouped[row[0], row[1], row[2]].append(row)
    sections = []
    for (version, page, key), parts in grouped.items():
        text = "\n\n".join(content.split("\n\n", 1)[-1] for *_, content, _ in parts)
        tokens = sum(r[6] for r in parts)
        sections.append(
            SectionText(version, page, key, tuple(parts[0][3]), parts[0][4], text, tokens)
        )
    return sections


def sample(
    sections: list[SectionText],
    n: int,
    seed: int,
    exclude: set[str] = frozenset(),
    min_tokens: int = MIN_TOKENS,
) -> list[SectionText]:
    """n random sections, taking one area at a time in turn so every area is represented."""
    by_area: dict[str, list[SectionText]] = defaultdict(list)
    for s in sorted(sections, key=lambda s: s.ref):
        if s.tokens >= min_tokens and s.ref not in exclude:
            by_area[area_for(s.page)].append(s)
    rng = random.Random(seed)
    for pool in by_area.values():
        rng.shuffle(pool)
    areas = sorted(by_area)
    rng.shuffle(areas)
    picked: list[SectionText] = []
    while len(picked) < n and any(by_area.values()):
        for area in areas:
            if by_area[area] and len(picked) < n:
                picked.append(by_area[area].pop())
    return picked


def _match(before: list[SectionText], after: list[SectionText]):
    """Pair sections across versions: same page and key, or the same key on a renamed page
    (18 moved the autovacuum settings to another page)."""
    after_by_ref = {(s.page, s.key): s for s in after}
    after_keys: dict[str, list[SectionText]] = defaultdict(list)
    for s in after:
        after_keys[s.key].append(s)
    before_keys: dict[str, int] = defaultdict(int)
    for s in before:
        before_keys[s.key] += 1
    pairs, matched = [], set()
    for s in before:
        twin = after_by_ref.get((s.page, s.key))
        if twin is None and before_keys[s.key] == 1 and len(after_keys[s.key]) == 1:
            twin = after_keys[s.key][0]
        if twin is not None:
            pairs.append((s, twin))
            matched.add(twin.ref)
    return pairs, [s for s in after if s.ref not in matched]


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE.split(text) if s.strip()]


def _comparable(sentence: str) -> str:
    return _XREF_NUMBER.sub(r"\1 N", sentence)


def changes(
    before: list[SectionText], after: list[SectionText], min_words: int = 8
) -> tuple[list[Change], list[SectionText]]:
    """Sections whose text really changed (at least min_words words, ignoring renumbered
    cross-references), biggest first; and sections that are new in `after`."""
    pairs, new = _match(before, after)
    found = []
    for a, b in pairs:
        old, cur = _sentences(a.text), _sentences(b.text)
        old_cmp, cur_cmp = [_comparable(x) for x in old], [_comparable(x) for x in cur]
        if old_cmp == cur_cmp:
            continue
        # Compare with cross-reference numbers masked, but report the original sentences, so
        # a quote copied from the output matches the docs.
        removed, added = [], []
        matcher = difflib.SequenceMatcher(a=old_cmp, b=cur_cmp, autojunk=False)
        for op, i1, i2, j1, j2 in matcher.get_opcodes():
            if op in ("delete", "replace"):
                removed += old[i1:i2]
            if op in ("insert", "replace"):
                added += cur[j1:j2]
        removed, added = tuple(removed), tuple(added)
        words = sum(len(s.split()) for s in removed + added)
        if words >= min_words:
            found.append(Change(a, b, removed, added, words))
    found.sort(key=lambda c: (-c.changed_words, c.after.ref))
    new = sorted((s for s in new if s.tokens >= MIN_TOKENS), key=lambda s: s.ref)
    return found, new
