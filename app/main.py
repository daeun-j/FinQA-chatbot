"""Main entry point: load components and launch the Gradio UI.

Builds BOTH oracle and retrieval graphs so the UI can toggle between them
per-question (oracle auto-enabled when user loads a curated/dev demo).
"""

import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    config_path = os.environ.get("CONFIG_PATH", "configs/config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    print("=== FinQA Chatbot ===")
    print(f"Model:    {config['model']['name']}")
    print(f"vLLM URL: {config['model']['vllm_base_url']}")

    print("\n[1/4] Connecting to vLLM server...")
    from src.llm.vllm_client import create_llm
    llm = create_llm(
        base_url=config["model"]["vllm_base_url"],
        model_name=config["model"]["name"],
        temperature=config["model"]["temperature"],
        max_tokens=config["model"]["max_tokens"],
    )

    print("[2/4] Loading retriever (FAISS + BM25 + reranker)...")
    retrieval_cfg = config["retrieval"]
    index_path = os.path.join(retrieval_cfg["index_path"], "index.faiss")
    required = [index_path, retrieval_cfg["chunks_path"], retrieval_cfg["bm25_path"]]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"Index files missing: {missing}. Run `python -m src.indexing.build_index` first."
        )
    reranker_cfg = config.get("reranker", {})
    reranker_model = reranker_cfg.get("model_name") if reranker_cfg.get("enabled", True) else None
    from src.retrieval.retriever import create_retriever
    retriever = create_retriever(
        index_path=index_path,
        docs_path=retrieval_cfg["chunks_path"],
        bm25_path=retrieval_cfg["bm25_path"],
        embedding_model=config["embedding"]["model_name"],
        reranker_model=reranker_model,
        top_k=retrieval_cfg["top_k"],
        dense_top_n=retrieval_cfg["dense_top_n"],
        bm25_top_n=retrieval_cfg["bm25_top_n"],
        rerank_top_n=retrieval_cfg["rerank_top_n"],
        rrf_k=retrieval_cfg["rrf_k"],
        aggregation=retrieval_cfg.get("aggregation", "first_seen"),
        count_top_n=retrieval_cfg.get("count_top_n", 20),
    )

    print("[3/4] Building LangGraph agents (retrieval + oracle, VERIFY disabled)...")
    from src.agent.graph import build_graph, build_oracle_graph
    retrieval_graph = build_graph(retriever, llm, use_verify=False)
    oracle_graph = build_oracle_graph(llm, use_verify=False)

    print("[4/4] Loading dev data for demo dropdown...")
    from src.data_processing.loader import load_finqa_file
    dev_path = os.path.join(config["data"]["raw_dir"], config["data"].get("dev_file", "dev.json"))
    dev_docs = load_finqa_file(dev_path) if os.path.exists(dev_path) else []
    print(f"  loaded {len(dev_docs)} dev examples")

    meta = {
        "model_name": config["model"]["name"],
        "vllm_base_url": config["model"]["vllm_base_url"],
        "use_verify": False,
        "retriever_top_k": retrieval_cfg["top_k"],
        "reranker": reranker_model or "(disabled)",
    }

    from app.gradio_ui import create_ui
    app = create_ui(
        retrieval_graph=retrieval_graph,
        oracle_graph=oracle_graph,
        retriever=retriever,
        dev_docs=dev_docs,
        meta=meta,
    )

    gradio_cfg = config.get("gradio", {})
    port = int(os.environ.get("FINQA_PORT", gradio_cfg.get("server_port", 7860)))
    share_env = os.environ.get("FINQA_SHARE")
    share = (share_env.lower() in ("1", "true", "yes")) if share_env else gradio_cfg.get("share", True)

    print(f"\nLaunching Gradio on port {port} (share={share})...")
    app.launch(server_name="0.0.0.0", server_port=port, share=share)


if __name__ == "__main__":
    main()
