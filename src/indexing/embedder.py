"""Embedding wrapper using SentenceTransformers."""

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


# BGE retrieval models use an asymmetric setup: a special instruction prefix on
# the query side, no prefix on the document side. Skipping this drops recall by
# ~half on bge-base-en-v1.5. Source: official BGE README.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    """Thin wrapper around SentenceTransformer for batch embedding."""

    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5", batch_size: int = 64):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)
        if device == "cuda":
            self.model.half()  # fp16 on GPU: ~2x faster, negligible quality drop for retrieval
        self.batch_size = batch_size
        self.model_name = model_name
        self.is_bge = "bge" in model_name.lower()
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"[embedder] {model_name} on {device} (batch_size={batch_size}, bge_prefix={self.is_bge})")

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed documents (no query prefix)."""
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        return np.array(embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a query, applying the BGE asymmetric instruction prefix."""
        text = (BGE_QUERY_PREFIX + query) if self.is_bge else query
        embedding = self.model.encode([text], normalize_embeddings=True)
        return np.array(embedding[0], dtype=np.float32)
