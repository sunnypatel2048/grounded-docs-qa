"""Pack each section's blocks into chunks that fit the embedding model and the reranker.

Rules:
- At most MAX_TOKENS (400) tokens per chunk, location line included. The embedding model
  takes 512, but a cross-encoder reranker reads the question and the chunk together in one
  512-token window, so 400 leaves room for the question.
- Chunks split only between blocks (paragraphs, code examples, table rows, list items...).
  A block bigger than the limit on its own is split between lines (code, fences repeated on
  each piece) or sentences (prose); a single line or sentence that is still too big is cut at
  token boundaries.
- Sizes are balanced: a 900-token section becomes two chunks of about 450 within the limit,
  never 400 + 400 + 100.
- Sections are never merged, however small: each has its own anchor, so each chunk can be
  cited with an exact link.
- Each chunk starts with its location (SELECT › Parameters › WITH Clause) but not the
  version. The version lives in its own column, so a section that is unchanged across
  versions produces identical text and the same content hash, and is embedded only once.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, replace

from docs_qa.parse import Page
from docs_qa.tokens import TokenCounter

MAX_TOKENS = 400
TRAIL_SEPARATOR = " › "
_SENTENCE_END = re.compile(r"(?<=[.!?:;])\s+|\n+")


@dataclass(frozen=True)
class Chunk:
    id: str  # "16/sql-select.html#SQL-WITH/0"
    version: str
    page_path: str
    section_key: str
    ordinal: int  # position within the section
    anchor: str | None
    url: str
    trail: tuple[str, ...]
    text: str  # location line + blank line + blocks; this is what gets embedded and searched
    token_count: int
    content_hash: str  # sha256 of text: same text in any version -> same hash -> one embedding


@dataclass
class ChunkStats:
    """How often oversized blocks needed splitting, and how."""

    split_code: int = 0  # code blocks split between lines
    split_prose: int = 0  # prose blocks split between sentences
    token_cuts: int = 0  # single lines or sentences cut at token boundaries


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Chunker:
    def __init__(self, counter: TokenCounter, max_tokens: int = MAX_TOKENS):
        if max_tokens < 50:
            raise ValueError("max_tokens must be at least 50")
        self.counter = counter
        self.max_tokens = max_tokens
        self.stats = ChunkStats()

    def chunk_page(self, page: Page) -> list[Chunk]:
        chunks: list[Chunk] = []
        for section in page.sections:
            header = self._header(section.trail)
            for ordinal, text in enumerate(self._chunk_texts(header, section.blocks)):
                chunks.append(
                    Chunk(
                        id=f"{page.version}/{page.path}#{section.key}/{ordinal}",
                        version=page.version,
                        page_path=page.path,
                        section_key=section.key,
                        ordinal=ordinal,
                        anchor=section.anchor,
                        url=section.url,
                        trail=section.trail,
                        text=text,
                        token_count=self.counter.count(text),
                        content_hash=content_hash(text),
                    )
                )
        return chunks

    def _header(self, trail: tuple[str, ...]) -> str:
        header = TRAIL_SEPARATOR.join(trail)
        limit = self.max_tokens // 4  # a long heading (a function signature) mustn't crowd out text
        if self.counter.count(header) > limit:
            header = self._cut_to_fit(header, limit)[0] + " …"
        return header

    def _chunk_texts(self, header: str, blocks: tuple[str, ...]) -> list[str]:
        budget = self.max_tokens - self.counter.count(header)
        stats_before = replace(self.stats)
        while True:
            self.stats = replace(stats_before)  # a retry mustn't count the same split twice
            pieces: list[tuple[str, int]] = []
            for block in blocks:
                tokens = self.counter.count(block)
                if tokens <= budget:
                    pieces.append((block, tokens))
                else:
                    pieces += self._split_block(block, budget)
            texts = [
                header + "\n\n" + "\n\n".join(text for text, _ in group)
                for group in balanced_groups(pieces, budget)
            ]
            # Counting pieces separately can differ by a token or two from counting the
            # joined text. Never exceed the limit: if that happens, pack a little tighter.
            if all(self.counter.count(t) <= self.max_tokens for t in texts) or budget <= 10:
                return texts
            budget -= max(budget // 20, 1)

    def _split_block(self, block: str, budget: int) -> list[tuple[str, int]]:
        if block.startswith("```\n") and block.endswith("\n```"):
            self.stats.split_code += 1
            fence_tokens = self.counter.count("```\n\n```")
            lines = self._fit_all(block[4:-4].split("\n"), budget - fence_tokens)
            groups = balanced_groups(lines, budget - fence_tokens)
            return self._measured(
                "```\n" + "\n".join(line for line, _ in group) + "\n```" for group in groups
            )
        self.stats.split_prose += 1
        sentences = self._fit_all([s for s in _SENTENCE_END.split(block) if s.strip()], budget)
        groups = balanced_groups(sentences, budget)
        return self._measured(" ".join(s for s, _ in group) for group in groups)

    def _fit_all(self, parts: list[str], budget: int) -> list[tuple[str, int]]:
        """Measure each part; cut any part that is still over budget at token boundaries."""
        out = []
        for part in parts:
            tokens = self.counter.count(part)
            if tokens <= budget:
                out.append((part, tokens))
            else:
                self.stats.token_cuts += 1
                out += self._measured(self._cut_to_fit(part, budget))
        return out

    def _cut_to_fit(self, text: str, budget: int) -> list[str]:
        size = budget
        while True:
            pieces = self.counter.cut(text, size)
            if all(self.counter.count(p) <= budget for p in pieces) or size == 1:
                return pieces
            size -= max(size // 10, 1)

    def _measured(self, texts) -> list[tuple[str, int]]:
        return [(t, self.counter.count(t)) for t in texts]


def balanced_groups(pieces: list[tuple[str, int]], budget: int) -> list[list[tuple[str, int]]]:
    """Group consecutive (text, tokens) pieces so each group fits the budget, using the fewest
    groups possible, with sizes as even as possible.

    Before each group: n = the fewest groups the remaining pieces need (filling greedily gives
    exactly that minimum when order is fixed), so aim for remaining tokens / n."""
    groups = []
    i = 0
    while i < len(pieces):
        remaining = sum(t for _, t in pieces[i:])
        target = remaining / _fewest_groups(pieces, i, budget)
        group, size = [pieces[i]], pieces[i][1]
        i += 1
        while i < len(pieces):
            tokens = pieces[i][1]
            if size + tokens > budget:
                break
            if size + tokens > target and (size + tokens - target) > (target - size):
                break  # stopping here lands closer to the target than taking this piece
            group.append(pieces[i])
            size += tokens
            i += 1
        groups.append(group)
    return groups


def _fewest_groups(pieces: list[tuple[str, int]], start: int, budget: int) -> int:
    count, size = 1, 0
    for _, tokens in pieces[start:]:
        if size and size + tokens > budget:
            count, size = count + 1, 0
        size += tokens
    return count
