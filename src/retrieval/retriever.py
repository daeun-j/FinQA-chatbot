"""Hybrid (dense + BM25) retriever with cross-encoder reranking.

Pipeline per query:

    1. Dense retrieval over chunks       -> top-N_dense chunks
    2. BM25 retrieval over chunks        -> top-N_bm25 chunks
    3. Reciprocal Rank Fusion            -> unified chunk ranking
    4. Cross-encoder rerank top-N_rerank -> sharpened chunk ranking
    5. Aggregate chunks to parent docs   -> top-k FinQADocuments

The agent graph consumes FinQADocuments (same interface as before), so the
upgrade is transparent to downstream nodes.
"""

from collections import defaultdict
from typing import Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from src.data_processing.chunk import Chunk
from src.data_processing.document import FinQADocument
from src.indexing.bm25_store import BM25Store
from src.indexing.embedder import Embedder
from src.indexing.faiss_store import FAISSStore
from src.retrieval.reranker import Reranker


def rrf_fuse(
    rankings: list[list[Chunk]],
    k: int = 60,
) -> list[tuple[Chunk, float]]:
    """Reciprocal Rank Fusion over multiple chunk rankings.

    Score = sum over rankings of 1 / (k + rank). Ties broken by the order they
    first appear in the input. Standard k=60 per Cormack et al. 2009.
    """
    scores: dict[str, float] = defaultdict(float)
    first_seen: dict[str, Chunk] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking):
            scores[chunk.chunk_id] += 1.0 / (k + rank + 1)
            first_seen.setdefault(chunk.chunk_id, chunk)
    ordered = sorted(scores.items(), key=lambda x: -x[1])
    return [(first_seen[cid], score) for cid, score in ordered]


def _finqa_doc_to_langchain(
    doc: FinQADocument,
    score: float,
    evidence_chunks: Optional[list[Chunk]] = None,
) -> Document:
    """Package a FinQADocument as a LangChain Document for the agent graph.

    Args:
        doc: The parent FinQA document.
        score: Top-chunk score for this document.
        evidence_chunks: The chunks (in rank order) that came from this doc.
            The agent uses these to surface row-level evidence hints.
    """
    evidence_payload = []
    for c in (evidence_chunks or []):
        evidence_payload.append({
            "chunk_id": c.chunk_id,
            "chunk_type": c.chunk_type,
            "row_index": c.row_index,
            "row_label": c.row_label,
            "text": c.text,
        })

    return Document(
        page_content=doc.get_context_for_llm(),
        metadata={
            "doc_id": doc.doc_id,
            "score": score,
            "table": doc.table,
            "table_md": doc.table_md,
            "pre_text": doc.pre_text,
            "post_text": doc.post_text,
            "question": doc.question,
            "gold_answer": doc.gold_answer,
            "gold_program": doc.gold_program,
            "evidence_chunks": evidence_payload,
        },
    )


class HybridRerankedRetriever(BaseRetriever):
    """Dense + BM25 + cross-encoder reranker, returning doc-level results."""

    dense_store: FAISSStore
    bm25_store: BM25Store
    embedder: Embedder
    reranker: Optional[Reranker] = None

    top_k: int = 3
    dense_top_n: int = 50
    bm25_top_n: int = 50
    rerank_top_n: int = 20
    rrf_k: int = 60
    aggregation: str = "first_seen"   # or "count"
    count_top_n: int = 20             # used when aggregation == "count"

    class Config:
        arbitrary_types_allowed = True

    def _dense_search(self, query: str) -> list[Chunk]:
        q_emb = self.embedder.embed_query(query)
        return [c for c, _ in self.dense_store.search(q_emb, top_k=self.dense_top_n)]

    def _bm25_search(self, query: str) -> list[Chunk]:
        return [c for c, _ in self.bm25_store.search(query, top_k=self.bm25_top_n)]

    def _aggregate_to_docs(
        self,
        ranked_chunks: list[tuple[Chunk, float]],
    ) -> list[tuple[FinQADocument, float, list[Chunk]]]:
        """Collapse chunks to their parent docs.

        Two aggregation strategies (controlled by self.aggregation):
        - "first_seen": doc score = its top chunk's score (rank-preserving).
        - "count":     doc score = how many of its chunks appear in the top
                       `count_top_n` ranked chunks. Best chunk's score breaks
                       ties. Helps when the right doc contributes multiple
                       relevant chunks (whole-table + matching rows + text).
        """
        best_score: dict[str, float] = {}
        first_doc: dict[str, FinQADocument] = {}
        evidence: dict[str, list[Chunk]] = {}
        first_order: list[str] = []
        for chunk, score in ranked_chunks:
            pid = chunk.parent_doc_id
            if pid not in best_score:
                best_score[pid] = score
                first_doc[pid] = chunk.parent_doc
                evidence[pid] = []
                first_order.append(pid)
            evidence[pid].append(chunk)

        if self.aggregation == "count":
            counted_chunks = ranked_chunks[: self.count_top_n]
            counts: dict[str, int] = {}
            for chunk, _ in counted_chunks:
                counts[chunk.parent_doc_id] = counts.get(chunk.parent_doc_id, 0) + 1
            # Sort by chunk count desc, then by best chunk score desc
            ordered_ids = sorted(
                counts.keys(),
                key=lambda pid: (-counts[pid], -best_score.get(pid, 0.0)),
            )
            # Append any docs that scored but didn't appear in top-N (unlikely)
            for pid in first_order:
                if pid not in counts:
                    ordered_ids.append(pid)
        else:  # "first_seen"
            ordered_ids = first_order

        return [(first_doc[pid], best_score[pid], evidence[pid]) for pid in ordered_ids]

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> list[Document]:
        dense_chunks = self._dense_search(query)
        bm25_chunks = self._bm25_search(query)

        fused = rrf_fuse([dense_chunks, bm25_chunks], k=self.rrf_k)

        if self.reranker is not None and fused:
            candidates = [c for c, _ in fused[: self.rerank_top_n]]
            reranked = self.reranker.rerank(query, candidates)
            ranked_chunks = reranked
        else:
            ranked_chunks = fused

        agg = self._aggregate_to_docs(ranked_chunks)[: self.top_k]
        return [
            _finqa_doc_to_langchain(doc, score, evidence_chunks=ev)
            for doc, score, ev in agg
        ]


def create_retriever(
    index_path: str,
    docs_path: str,
    bm25_path: str,
    embedding_model: str = "BAAI/bge-base-en-v1.5",
    reranker_model: Optional[str] = "BAAI/bge-reranker-v2-m3",
    top_k: int = 3,
    dense_top_n: int = 50,
    bm25_top_n: int = 50,
    rerank_top_n: int = 20,
    rrf_k: int = 60,
    aggregation: str = "first_seen",
    count_top_n: int = 20,
) -> HybridRerankedRetriever:
    """Factory: load chunk-level FAISS + BM25 stores and assemble the stack.

    Args:
        index_path: Path to the FAISS index (built over chunks).
        docs_path: Path to the pickled chunk list aligned with the FAISS index.
        bm25_path: Path to the pickled BM25 store.
        embedding_model: Bi-encoder name (must match the one used at indexing).
        reranker_model: Cross-encoder name, or None to skip reranking.
    """
    dense_store = FAISSStore.load(index_path, docs_path)
    bm25_store = BM25Store.load(bm25_path)
    embedder = Embedder(model_name=embedding_model)
    reranker = Reranker(model_name=reranker_model) if reranker_model else None

    return HybridRerankedRetriever(
        dense_store=dense_store,
        bm25_store=bm25_store,
        embedder=embedder,
        reranker=reranker,
        top_k=top_k,
        dense_top_n=dense_top_n,
        bm25_top_n=bm25_top_n,
        rerank_top_n=rerank_top_n,
        rrf_k=rrf_k,
        aggregation=aggregation,
        count_top_n=count_top_n,
    )
