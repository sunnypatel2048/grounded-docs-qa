"""Embedding models: the real one, and a deterministic stand-in for tests.

The real model runs locally with Hugging Face transformers and torch, on the Mac's GPU (Apple
MPS) when available, following the model card's recipe: the [CLS] token's final hidden state,
normalized. torch is large, so it lives in the `embed` dependency group, which CI leaves out;
tests use HashEmbedder.

Vectors are normalized to length 1, so cosine distance (the HNSW index's vector_cosine_ops)
is the right comparison.

Note for search (Task 10): bge models expect QUERY_INSTRUCTION in front of a *question*
before embedding it. Stored passages are embedded as they are, with no prefix.
"""

from __future__ import annotations

import hashlib
import math
import struct
from typing import Protocol

from docs_qa.tokens import EMBEDDING_MODEL

EMBEDDING_DIM = 384  # must match vector(384) in the schema
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    model: str
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]:
        """One normalized vector per text, in order."""
        ...


def best_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class LocalEmbedder:
    def __init__(self, model: str = EMBEDDING_MODEL, device: str | None = None, batch_size=32):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self.model = model
        self.device = device or best_device()
        self.batch_size = batch_size
        self._tokenizer = AutoTokenizer.from_pretrained(model)
        self._model = AutoModel.from_pretrained(model).to(self.device).eval()
        self.dim = self._model.config.hidden_size

    def embed(self, texts: list[str]) -> list[list[float]]:
        torch = self._torch
        # Batch texts of similar length together so little compute goes on padding.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        vectors: list[list[float] | None] = [None] * len(texts)
        for start in range(0, len(order), self.batch_size):
            indexes = order[start : start + self.batch_size]
            encoded = self._tokenizer(
                [texts[i] for i in indexes],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                cls = self._model(**encoded).last_hidden_state[:, 0]
            normalized = torch.nn.functional.normalize(cls, p=2, dim=1).cpu().tolist()
            for i, vector in zip(indexes, normalized, strict=True):
                vectors[i] = vector
        return vectors  # type: ignore[return-value]


class HashEmbedder:
    """Deterministic fake vectors derived from the text's hash. Same text, same vector.
    Records every text it embeds, so tests can check what was (not) embedded."""

    model = "hash-embedder"

    def __init__(self, dim: int = EMBEDDING_DIM):
        self.dim = dim
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self._vector(t) for t in texts]

    @property
    def embedded(self) -> list[str]:
        return [t for call in self.calls for t in call]

    def _vector(self, text: str) -> list[float]:
        raw = b""
        counter = 0
        while len(raw) < self.dim * 4:
            raw += hashlib.sha256(f"{counter}:{text}".encode()).digest()
            counter += 1
        values = [v / 2**32 - 0.5 for v in struct.unpack(f"<{self.dim}I", raw[: self.dim * 4])]
        norm = math.sqrt(sum(v * v for v in values))
        return [v / norm for v in values]
