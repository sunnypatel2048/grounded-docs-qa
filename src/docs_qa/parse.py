"""Turn a saved documentation page into sections of clean text blocks.

A page's real content is the single div#docContent; the site menu, version switcher, search box
and footer around it are dropped, as are the navigation header and footer, table-of-contents
lists and the "#" permalink icons.

Sections come from the DocBook section divs (sect1, sect2, refsect1, ...), never from heading tags
alone: note boxes carry their own <h3>Note</h3>, and a heading-based splitter would turn every
note into a fake section. Settings, functions, psql commands and glossary terms are written as
definition-list entries with their own IDs (<dt id="GUC-AUTOVACUUM-NAPTIME">); each of those
becomes its own section so it can be retrieved and cited on its own.

Each section has two identifiers:
- anchor: the element's real HTML id, used to link to it (works for that release).
- key: a stable identifier for chunk IDs. Hand-written ids (SQL-WITH, GUC-...) are used as they
  are. Generated ids (id-1.9.3.172.7) encode the section's position in the book and shift when
  pages are added, so those sections are keyed by their parent's key plus their heading
  (SQL-SELECT/description) instead.

A section's text is a list of blocks, each a unit that must not be split: a paragraph, a code
example, a table row, a list item, a definition, a note. The chunker only splits between them.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from bs4 import BeautifulSoup, Comment, NavigableString, Tag

from docs_qa.fetch import BASE_URL

SECTION_CLASSES = frozenset(
    {
        "book", "part", "reference", "chapter", "appendix", "preface", "bibliography",
        "glossary", "refentry", "refsynopsisdiv", "refsect1", "refsect2", "refsect3",
        "sect1", "sect2", "sect3", "sect4", "sect5", "simplesect", "section",
    }
)  # fmt: skip
ADMONITIONS = frozenset({"note", "tip", "warning", "caution", "important"})
DROPPED_CLASSES = frozenset({"navheader", "navfooter", "toc", "lot", "indexterm", "id_link"})
SKIPPED_KINDS = frozenset({"index"})  # the alphabetical book index: thousands of bare links
HEADINGS = ("h1", "h2", "h3", "h4", "h5", "h6")
BLOCK_TAGS = frozenset(
    {"p", "pre", "div", "table", "dl", "dt", "dd", "ul", "ol", "li", "blockquote", "hr", *HEADINGS}
)
GENERATED_ID_PREFIXES = ("id-", "ientry-", "idm")

_TITLE_PREFIX = re.compile(r"^PostgreSQL: Documentation: [^:]+:\s*")
_RELEASE = re.compile(r"PostgreSQL (\d+\.\d+) Documentation")
# "5.1. ", "Chapter 8. ", "Part II. ", "F.3.2. ", "Table 9.9. " style numbering
_NUMBER = re.compile(r"^(?:(?:Chapter|Part|Appendix)\s+)?(?:\d+|[A-Z]|[IVXLC]+)(?:\.\d+)*\.\s+")


class ParseError(Exception):
    """The page doesn't look like a PostgreSQL documentation page."""


@dataclass(frozen=True)
class Section:
    key: str  # stable within the page, e.g. "SQL-SELECT/description" or "GUC-AUTOVACUUM-NAPTIME"
    anchor: str | None  # real HTML id, for links
    heading: str  # as printed, e.g. "5.1. Identifying Bugs"
    trail: tuple[str, ...]  # headings from the page down to here, numbering removed
    url: str
    blocks: tuple[str, ...]


@dataclass(frozen=True)
class Page:
    version: str
    path: str
    url: str
    title: str
    release: str | None  # exact docs release, e.g. "16.15"
    kind: str  # DocBook element of the page: sect1, refentry, chapter, ...
    sections: tuple[Section, ...] = field(default=())


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def strip_number(heading: str) -> str:
    return _NUMBER.sub("", heading).strip() or heading


def is_stable_id(value: str | None) -> bool:
    return bool(value) and not value.startswith(GENERATED_ID_PREFIXES)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "section"


def _classes(el: Tag) -> set[str]:
    return set(el.get("class") or ())


# --- page -----------------------------------------------------------------------------------


def parse_page(html: str, version: str, path: str) -> Page:
    soup = BeautifulSoup(html, "html.parser")
    content = soup.find("div", id="docContent")
    if content is None:
        raise ParseError(f"{version}/{path}: no div#docContent")

    page_title = normalize(soup.title.get_text()) if soup.title else path
    page_title = _TITLE_PREFIX.sub("", page_title)
    release = _release(content, soup, version)

    for comment in content.find_all(string=lambda s: isinstance(s, Comment)):
        comment.extract()
    for el in content.find_all(True):
        if not el.decomposed and _classes(el) & DROPPED_CLASSES:  # skip children of removed
            el.decompose()

    root = next((c for c in content.children if isinstance(c, Tag)), None)
    if root is None:
        raise ParseError(f"{version}/{path}: div#docContent is empty")
    kind = (root.get("class") or ["unknown"])[0]
    url = f"{BASE_URL}/docs/{version}/{path}"
    page = Page(version, path, url, page_title, release, kind)
    if kind in SKIPPED_KINDS:
        return page

    walker = _Walker(url)
    walker.section(root, trail=(), parent_key="", fallback_heading=page_title)
    sections = tuple(s for s in walker.sections if s.blocks)
    return Page(version, path, url, page_title, release, kind, sections)


def _release(content: Tag, soup: BeautifulSoup, version: str) -> str | None:
    home = content.find("a", accesskey="h")
    candidates = [
        home.get("title", "") if home else "",
        soup.title.get_text() if soup.title else "",
    ]
    for text in candidates:
        m = _RELEASE.search(text.replace("\xa0", " "))
        if m and m.group(1).split(".")[0] == version:
            return m.group(1)
    return None


def parse_file(raw_dir: Path, version: str, name: str) -> Page:
    html = (Path(raw_dir) / version / name).read_text(encoding="utf-8")
    return parse_page(html, version, name)


def iter_page_names(raw_dir: Path, version: str) -> Iterator[str]:
    yield from sorted(p.name for p in (Path(raw_dir) / version).glob("*.html"))


# --- sections and blocks --------------------------------------------------------------------


class _Walker:
    def __init__(self, page_url: str):
        self.page_url = page_url
        self.sections: list[Section] = []
        self._used_keys: Counter[str] = Counter()

    def _key(self, anchor: str | None, parent_key: str, title: str) -> str:
        if is_stable_id(anchor):
            base = anchor
        else:
            base = f"{parent_key}/{_slug(title)}" if parent_key else _slug(title)
        self._used_keys[base] += 1
        n = self._used_keys[base]
        return base if n == 1 else f"{base}-{n}"

    def _add(
        self, anchor: str | None, heading: str, trail: tuple[str, ...], parent_key: str, body: Tag
    ) -> None:
        """Record one section, then the sections found inside it (so parents come first)."""
        title = strip_number(heading)
        trail = (*trail, title)
        key = self._key(anchor, parent_key, title)
        url = f"{self.page_url}#{anchor}" if anchor else self.page_url
        pending: list[Tag | tuple[list[Tag], Tag]] = []
        blocks = self.blocks(body, pending)
        self.sections.append(Section(key, anchor, heading, trail, url, tuple(blocks)))
        for item in pending:
            if isinstance(item, Tag):  # a nested section div
                self.section(item, trail, key)
            else:  # a definition-list entry with its own id: its terms and its definition
                terms, definition = item
                term_anchor = next(t["id"] for t in terms if is_stable_id(t.get("id")))
                term = ", ".join(normalize(t.get_text()) for t in terms)
                self._add(term_anchor, term, trail, key, definition)

    def section(
        self, el: Tag, trail: tuple[str, ...], parent_key: str, fallback_heading: str = ""
    ) -> None:
        heading_el = _heading_element(el)
        heading = normalize(heading_el.get_text()) if heading_el else ""
        if heading_el is not None:
            heading_el.extract()
        self._add(el.get("id"), heading or fallback_heading or "section", trail, parent_key, el)

    # Blocks. `pending` collects nested sections, which are recorded after the current one.

    def blocks(self, container: Tag, pending: list) -> list[str]:
        """The blocks inside a container, treating inline content as paragraphs."""
        out: list[str] = []
        buffer: list[str] = []

        def flush() -> None:
            text = normalize("".join(buffer))
            if text:
                out.append(text)
            buffer.clear()

        for child in container.children:
            if isinstance(child, NavigableString):
                buffer.append(str(child))
            elif isinstance(child, Tag):
                if child.name in BLOCK_TAGS:
                    flush()
                    out.extend(self.block(child, pending))
                else:
                    buffer.append(child.get_text())
        flush()
        return out

    def block(self, el: Tag, pending: list) -> list[str]:
        classes = _classes(el)
        if el.name == "div" and classes & SECTION_CLASSES:
            pending.append(el)
            return []
        if el.name == "hr":
            return []
        if el.name == "pre":
            code = el.get_text().replace("\xa0", " ").strip("\n")
            return [f"```\n{code}\n```"] if code.strip() else []
        if el.name == "table":
            return self.table(el)
        if el.name == "dl":
            return self.definitions(el, pending)
        if el.name in ("ul", "ol"):
            return self.list_items(el, pending)
        if classes & ADMONITIONS:
            first = next((c for c in el.children if isinstance(c, Tag)), None)
            label = sorted(classes & ADMONITIONS)[0].capitalize()
            if first is not None and first.name in HEADINGS:
                label = normalize(first.get_text()) or label
                first.extract()
            inner = self.blocks(el, pending)
            return [f"{label}: " + "\n".join(inner)] if inner else []
        if el.name in ("p", *HEADINGS) and not el.find(BLOCK_TAGS - {"p"}):
            text = normalize(el.get_text())
            return [text] if text else []
        return self.blocks(el, pending)  # any other container: look inside

    def list_items(self, el: Tag, pending: list) -> list[str]:
        out = []
        number = 0
        for li in el.find_all("li", recursive=False):
            inner = self.blocks(li, pending)
            if not inner:
                continue
            number += 1
            marker = f"{number}." if el.name == "ol" else "-"
            text = "\n".join(inner).replace("\n", "\n  ")
            out.append(f"{marker} {text}")
        return out

    def definitions(self, el: Tag, pending: list) -> list[str]:
        out: list[str] = []
        terms: list[Tag] = []
        for child in el.children:
            if not isinstance(child, Tag):
                continue
            if child.name == "dt":
                terms.append(child)
            elif child.name == "dd":
                if any(is_stable_id(t.get("id")) for t in terms):
                    pending.append((terms, child))
                else:
                    term = ", ".join(normalize(t.get_text()) for t in terms)
                    body = "\n".join(self.blocks(child, pending))
                    text = f"{term}: {body}" if term and body else term or body
                    if text:
                        out.append(text)
                terms = []
        out.extend(t for t in (normalize(t.get_text()) for t in terms) if t)  # dt without dd
        return out

    def table(self, el: Tag) -> list[str]:
        """One block per row. Multi-column rows are labelled with their column headers, so a
        row still makes sense when the chunker puts it in a different chunk from the header."""
        head_rows, body_rows = _table_rows(el)
        header = [self.cell(c) for c in _cells(head_rows[0])] if head_rows else []
        out = []
        for tr in body_rows:
            cells = [self.cell(c) for c in _cells(tr)]
            if len(cells) == 1:
                text = cells[0]
            elif len(header) == len(cells):
                pairs = zip(header, cells, strict=True)
                text = " | ".join(f"{h}: {c}" if h else c for h, c in pairs if c)
            else:
                text = " | ".join(c for c in cells if c)
            if text:
                out.append(text)
        return out

    def cell(self, el: Tag) -> str:
        return "\n".join(self.blocks(el, pending=[]))


def _table_rows(table: Tag) -> tuple[list[Tag], list[Tag]]:
    """(header rows, body rows), without descending into tables nested in cells."""
    head: list[Tag] = []
    body: list[Tag] = []
    for group in table.find_all(["thead", "tbody", "tfoot", "tr"], recursive=False):
        if group.name == "tr":
            body.append(group)
        else:
            (head if group.name == "thead" else body).extend(group.find_all("tr", recursive=False))
    return head, body


def _cells(row: Tag) -> list[Tag]:
    return row.find_all(["td", "th"], recursive=False)


def _heading_element(section: Tag) -> Tag | None:
    """The section's own title: in its titlepage, directly inside it, or (for reference pages)
    in its refnamediv. Never a heading inside a note, example or nested section."""
    for child in section.children:
        if not isinstance(child, Tag):
            continue
        if child.name in HEADINGS:
            return child
        classes = _classes(child)
        if "legalnotice-title" in classes:
            return child
        if classes & {"titlepage", "refnamediv"}:
            found = child.find(HEADINGS)
            if found is not None:
                return found
            continue  # an empty titlepage comes before a reference page's refnamediv
        return None
    return None
