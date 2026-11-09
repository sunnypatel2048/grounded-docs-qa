"""Token counting with the embedding model's own tokenizer.

Chunk sizes have to be measured in the model's tokens, not words or characters: the model
silently ignores everything past its limit. Only the tokenizer is needed here, through the small
`tokenizers` library (no torch); it downloads the model's tokenizer.json once and caches it.

Tests pass a WhitespaceCounter instead, so they never download anything.
"""

from __future__ import annotations

from typing import Protocol

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # 384 dimensions, 512-token input limit


class TokenCounter(Protocol):
    def count(self, text: str) -> int:
        """Tokens in text, not counting the model's [CLS]/[SEP] markers."""
        ...

    def cut(self, text: str, max_tokens: int) -> list[str]:
        """Split text at token boundaries into pieces of at most max_tokens each."""
        ...


class HFTokenCounter:
    def __init__(self, model: str = EMBEDDING_MODEL):
        from tokenizers import Tokenizer

        self.model = model
        self._tokenizer = Tokenizer.from_pretrained(model)
        # The saved tokenizer may truncate at 512 tokens, which would make every long text
        # look like it fits. Counting needs the full length.
        self._tokenizer.no_truncation()
        self._tokenizer.no_padding()

    def count(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False).ids)

    def cut(self, text: str, max_tokens: int) -> list[str]:
        offsets = self._tokenizer.encode(text, add_special_tokens=False).offsets
        pieces = []
        for start in range(0, len(offsets), max_tokens):
            window = offsets[start : start + max_tokens]
            end = offsets[start + max_tokens][0] if start + max_tokens < len(offsets) else None
            piece = text[window[0][0] : end].strip()
            if piece:
                pieces.append(piece)
        return pieces


class WhitespaceCounter:
    """One token per whitespace-separated word. For tests and offline experiments only."""

    model = "whitespace"

    def count(self, text: str) -> int:
        return len(text.split())

    def cut(self, text: str, max_tokens: int) -> list[str]:
        words = text.split()
        return [" ".join(words[i : i + max_tokens]) for i in range(0, len(words), max_tokens)]
