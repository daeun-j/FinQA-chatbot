"""BM25 sparse retrieval store over Chunks.

Complements the dense FAISS store: BM25 catches exact-match signal from
financial line items ("operating cash flow", "goodwill impairment") and
company names that dense embeddings often miss.
"""

import os
import pickle
import re
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi

from src.data_processing.chunk import Chunk


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[.\-][A-Za-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lowercase tokenization that keeps numbers and hyphenated terms intact.

    Keeping "10-k", "2.5", "eps" as single tokens matters for financial text.
    """
    return _TOKEN_RE.findall(text.lower())


class BM25Store:
    """Persistable BM25Okapi over a Chunk corpus."""

    def __init__(self):
        self.bm25: Optional[BM25Okapi] = None
        self.chunks: list[Chunk] = []

    def build(self, chunks: list[Chunk]):
        tokenized = [tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(tokenized)
        self.chunks = chunks

    def search(self, query: str, top_k: int) -> list[tuple[Chunk, float]]:
        assert self.bm25 is not None, "BM25Store not built or loaded"
        scores = self.bm25.get_scores(tokenize(query))
        if len(scores) == 0:
            return []
        top_k = min(top_k, len(scores))
        top_idx = np.argpartition(-scores, top_k - 1)[:top_k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [(self.chunks[i], float(scores[i])) for i in top_idx]

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "chunks": self.chunks}, f)
        print(f"[saved] BM25 store ({len(self.chunks)} chunks) -> {path}")

    @classmethod
    def load(cls, path: str) -> "BM25Store":
        with open(path, "rb") as f:
            payload = pickle.load(f)
        store = cls()
        store.bm25 = payload["bm25"]
        store.chunks = payload["chunks"]
        print(f"[loaded] BM25 store ({len(store.chunks)} chunks)")
        return store
