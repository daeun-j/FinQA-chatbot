"""Measure retrieval quality (recall@k) on a FinQA split.

Each FinQA example carries its own doc_id, so a perfect retriever returns
that exact doc when queried with the example's question. We compute
recall@1/3/5/10 and break down failures by chunk type.
"""

import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing.loader import load_finqa_file
from src.indexing.bm25_store import BM25Store
from src.indexing.embedder import Embedder
from src.indexing.faiss_store import FAISSStore
from src.retrieval.reranker import Reranker
from src.retrieval.retriever import rrf_fuse


def measure(config_path: str, split: str, max_examples: int, k_values: list[int]):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    rcfg = config["retrieval"]
    dense = FAISSStore.load(os.path.join(rcfg["index_path"], "index.faiss"), rcfg["chunks_path"])
    bm25 = BM25Store.load(rcfg["bm25_path"])
    embedder = Embedder(model_name=config["embedding"]["model_name"])
    reranker_cfg = config.get("reranker", {})
    reranker = Reranker(reranker_cfg["model_name"]) if reranker_cfg.get("enabled") else None

    data_path = os.path.join(config["data"]["raw_dir"], f"{split}.json")
    examples = load_finqa_file(data_path)
    if max_examples:
        examples = examples[:max_examples]

    max_k = max(k_values)
    n_dense = max(rcfg["dense_top_n"], max_k * 4)
    n_bm25 = max(rcfg["bm25_top_n"], max_k * 4)
    n_rerank = max(rcfg["rerank_top_n"], max_k * 2)

    # Stages we measure independently
    dense_only_hits = {k: 0 for k in k_values}
    bm25_only_hits = {k: 0 for k in k_values}
    rrf_hits = {k: 0 for k in k_values}
    rerank_hits = {k: 0 for k in k_values}
    rerank_count_hits = {k: 0 for k in k_values}  # rerank + count aggregation

    for i, ex in enumerate(examples):
        gold_id = ex.doc_id
        q = ex.question

        q_emb = embedder.embed_query(q)
        d_chunks = [c for c, _ in dense.search(q_emb, top_k=n_dense)]
        b_chunks = [c for c, _ in bm25.search(q, top_k=n_bm25)]

        # Collapse chunk lists to ordered unique parent_doc_ids
        def doc_ids(chunks):
            seen, out = set(), []
            for c in chunks:
                if c.parent_doc_id not in seen:
                    seen.add(c.parent_doc_id)
                    out.append(c.parent_doc_id)
            return out

        dense_doc_rank = doc_ids(d_chunks)
        bm25_doc_rank = doc_ids(b_chunks)

        fused = rrf_fuse([d_chunks, b_chunks], k=rcfg["rrf_k"])
        rrf_doc_rank = doc_ids([c for c, _ in fused])

        if reranker:
            cand = [c for c, _ in fused[:n_rerank]]
            reranked = reranker.rerank(q, cand)
            rerank_chunks = [c for c, _ in reranked]
            rerank_doc_rank = doc_ids(rerank_chunks)
            # Count-aggregation alternative: doc score = #chunks in top-20
            from collections import Counter
            counts = Counter(c.parent_doc_id for c in rerank_chunks[:rcfg.get("count_top_n", 20)])
            count_doc_rank = [pid for pid, _ in counts.most_common()]
            for pid in rerank_doc_rank:
                if pid not in count_doc_rank:
                    count_doc_rank.append(pid)
        else:
            rerank_doc_rank = rrf_doc_rank
            count_doc_rank = rrf_doc_rank

        for k in k_values:
            if gold_id in dense_doc_rank[:k]:
                dense_only_hits[k] += 1
            if gold_id in bm25_doc_rank[:k]:
                bm25_only_hits[k] += 1
            if gold_id in rrf_doc_rank[:k]:
                rrf_hits[k] += 1
            if gold_id in rerank_doc_rank[:k]:
                rerank_hits[k] += 1
            if gold_id in count_doc_rank[:k]:
                rerank_count_hits[k] += 1

        if (i + 1) % 25 == 0:
            print(f"  processed {i+1}/{len(examples)}")

    n = len(examples)
    print(f"\n=== Recall@k on {split} ({n} examples) ===")
    print(f"{'k':>4} | {'dense':>8} | {'bm25':>8} | {'+RRF':>8} | {'+rerank':>8} | {'+count':>8}")
    print("-" * 62)
    for k in k_values:
        print(
            f"{k:>4} | "
            f"{dense_only_hits[k]/n:>8.3f} | "
            f"{bm25_only_hits[k]/n:>8.3f} | "
            f"{rrf_hits[k]/n:>8.3f} | "
            f"{rerank_hits[k]/n:>8.3f} | "
            f"{rerank_count_hits[k]/n:>8.3f}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--max-examples", type=int, default=200)
    ap.add_argument("--k", default="1,3,5,10")
    args = ap.parse_args()

    k_values = [int(x) for x in args.k.split(",")]
    measure(args.config, args.split, args.max_examples, k_values)


if __name__ == "__main__":
    main()
