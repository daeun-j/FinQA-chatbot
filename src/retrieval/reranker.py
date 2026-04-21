"""Cross-encoder reranker for the retrieval pipeline.

A bi-encoder (FAISS + sentence-transformers) gives cheap recall, but scores
query and document independently. A cross-encoder reads them jointly and
produces a much sharper relevance signal. We use it to rerank the top-N
candidates from the hybrid (dense + BM25) stage.
"""

from typing import Optional

from sentence_transformers import CrossEncoder

from src.data_processing.chunk import Chunk


class Reranker:
    """Thin wrapper around a HuggingFace cross-encoder."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", max_length: int = 512):
        self.model = CrossEncoder(model_name, max_length=max_length)

    def rerank(
        self,
        query: str,
        chunks: list[Chunk],
        top_k: Optional[int] = None,
    ) -> list[tuple[Chunk, float]]:
        """Score (query, chunk.text) pairs and return chunks sorted by score.

        Args:
            query: User question.
            chunks: Candidate chunks from the retriever.
            top_k: Return only the top_k after reranking (None = all).
        """
        if not chunks:
            return []
        pairs = [(query, c.text) for c in chunks]
        scores = self.model.predict(pairs, show_progress_bar=False)
        scored = sorted(zip(chunks, scores), key=lambda x: -float(x[1]))
        if top_k is not None:
            scored = scored[:top_k]
        return [(c, float(s)) for c, s in scored]
