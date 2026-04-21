"""FAISS index for embedding-based retrieval.

Generic over the stored item type: we now index Chunks (post chunking
refactor), but the store doesn't care — callers pass any picklable objects
aligned row-wise with the embeddings.
"""

import os
import pickle
from typing import Any, Optional

import faiss
import numpy as np


class FAISSStore:
    """FAISS-based vector store.

    Uses IndexFlatIP (exact inner product search) since the corpus
    is small (~tens of thousands of chunks). With normalized embeddings
    this is equivalent to cosine similarity.
    """

    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.documents: list[Any] = []

    def add(self, embeddings: np.ndarray, documents: list[Any]):
        """Add items and their embeddings to the index.

        Args:
            embeddings: numpy array of shape (n, dimension).
            documents: Corresponding items (Chunks or FinQADocuments).
        """
        assert len(embeddings) == len(documents), (
            f"Mismatch: {len(embeddings)} embeddings vs {len(documents)} documents"
        )
        assert embeddings.shape[1] == self.dimension, (
            f"Dimension mismatch: expected {self.dimension}, got {embeddings.shape[1]}"
        )
        self.index.add(embeddings)
        self.documents.extend(documents)

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> list[tuple[Any, float]]:
        """Search for the most similar documents.

        Args:
            query_embedding: Query vector of shape (dimension,).
            top_k: Number of results to return.

        Returns:
            List of (document, score) tuples, sorted by descending similarity.
        """
        query = query_embedding.reshape(1, -1)
        scores, indices = self.index.search(query, top_k)

        results: list[tuple[Any, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents) and idx >= 0:
                results.append((self.documents[idx], float(score)))
        return results

    def save(self, index_path: str, docs_path: str):
        """Persist the FAISS index and aligned item list to disk."""
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        os.makedirs(os.path.dirname(docs_path), exist_ok=True)

        faiss.write_index(self.index, index_path)
        with open(docs_path, "wb") as f:
            pickle.dump(self.documents, f)

        print(f"[saved] FAISS index ({self.index.ntotal} vectors) -> {index_path}")
        print(f"[saved] Documents ({len(self.documents)}) -> {docs_path}")

    @classmethod
    def load(cls, index_path: str, docs_path: str) -> "FAISSStore":
        """Load a persisted FAISS store from disk.

        Args:
            index_path: Path to the FAISS index file.
            docs_path: Path to the pickled documents file.

        Returns:
            FAISSStore instance with loaded index and documents.
        """
        index = faiss.read_index(index_path)
        with open(docs_path, "rb") as f:
            documents = pickle.load(f)

        store = cls(dimension=index.d)
        store.index = index
        store.documents = documents

        print(f"[loaded] FAISS index ({store.index.ntotal} vectors)")
        print(f"[loaded] Documents ({len(store.documents)})")
        return store
