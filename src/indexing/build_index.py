"""Build chunk-level FAISS and BM25 indexes from FinQA dataset."""

import os
import pickle

import yaml

from src.data_processing.chunker import chunk_documents
from src.data_processing.loader import load_finqa_file
from src.indexing.bm25_store import BM25Store
from src.indexing.embedder import Embedder
from src.indexing.faiss_store import FAISSStore


def build_index(config_path: str = "configs/config.yaml"):
    """Chunk the corpus, build dense (FAISS) + sparse (BM25) indexes, save all."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    data_dir = config["data"]["raw_dir"]
    train_path = os.path.join(data_dir, config["data"]["train_file"])
    dev_path = os.path.join(data_dir, config["data"]["dev_file"])

    print("=== Loading documents ===")
    documents = load_finqa_file(train_path)
    if os.path.exists(dev_path):
        documents.extend(load_finqa_file(dev_path))
    print(f"Total documents: {len(documents)}")

    print("\n=== Chunking documents ===")
    min_chars = config.get("chunking", {}).get("min_chars", 120)
    chunks = chunk_documents(documents, min_chars=min_chars)
    print(f"Total chunks: {len(chunks)} (avg {len(chunks) / max(len(documents), 1):.1f} per doc)")

    print("\n=== Embedding chunks ===")
    embedder = Embedder(
        model_name=config["embedding"]["model_name"],
        batch_size=config["embedding"]["batch_size"],
    )
    embeddings = embedder.embed([c.text for c in chunks])
    print(f"Embeddings shape: {embeddings.shape}")

    print("\n=== Building FAISS index ===")
    dense_store = FAISSStore(dimension=config["embedding"]["dimension"])
    dense_store.add(embeddings, chunks)

    index_path = os.path.join(config["retrieval"]["index_path"], "index.faiss")
    chunks_path = config["retrieval"]["chunks_path"]
    dense_store.save(index_path, chunks_path)

    print("\n=== Building BM25 index ===")
    bm25_store = BM25Store()
    bm25_store.build(chunks)
    bm25_store.save(config["retrieval"]["bm25_path"])

    print("\n=== Saving parent document list ===")
    docs_path = config["retrieval"]["docs_path"]
    os.makedirs(os.path.dirname(docs_path), exist_ok=True)
    with open(docs_path, "wb") as f:
        pickle.dump(documents, f)
    print(f"[saved] Parent documents ({len(documents)}) -> {docs_path}")

    print("\n=== Index build complete ===")


if __name__ == "__main__":
    build_index()
