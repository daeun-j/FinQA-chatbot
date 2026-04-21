"""CLI entry point for evaluation."""

import os
import sys
import yaml
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm.vllm_client import create_llm
from src.retrieval.retriever import create_retriever
from src.agent.graph import build_graph, build_oracle_graph
from src.evaluation.runner import run_evaluation


def main():
    parser = argparse.ArgumentParser(description="Evaluate FinQA agent")
    parser.add_argument("--config", default="configs/config.yaml", help="Config path")
    parser.add_argument("--split", default="dev", help="Dataset split (dev/test)")
    parser.add_argument("--max-examples", type=int, default=None, help="Limit examples")
    parser.add_argument("--output", default="results/eval_results.json", help="Output path")
    parser.add_argument("--oracle", action="store_true",
                        help="Skip retrieval; feed the gold doc directly to the agent.")
    parser.add_argument("--n-samples", type=int, default=1,
                        help="Self-consistency: sample N times at higher temperature, vote.")
    parser.add_argument("--sc-temperature", type=float, default=0.7,
                        help="LLM temperature when n-samples > 1.")
    parser.add_argument("--baseline", action="store_true",
                        help="Use vanilla single-call RAG (no LangGraph, no calculator tool).")
    parser.add_argument("--no-verify", action="store_true",
                        help="Disable the VERIFY critic node in the LangGraph agent.")
    parser.add_argument("--dynamic-few-shot", action="store_true",
                        help="Replace static few-shot with top-K similar train (Q, gold_program) examples.")
    parser.add_argument("--num-few-shot", type=int, default=3,
                        help="Number of few-shot examples (static or dynamic).")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Build components
    print("=== Setting up ===")
    # Self-consistency requires sample diversity; bump temperature when n>1.
    llm_temperature = args.sc_temperature if args.n_samples > 1 else config["model"]["temperature"]
    if args.n_samples > 1:
        print(f"[self-consistency] n_samples={args.n_samples}, temperature={llm_temperature}")
    llm = create_llm(
        base_url=config["model"]["vllm_base_url"],
        model_name=config["model"]["name"],
        temperature=llm_temperature,
        max_tokens=config["model"]["max_tokens"],
    )

    # Retriever is needed unless oracle mode (and even baseline uses it).
    retriever = None
    if not args.oracle:
        index_path = os.path.join(config["retrieval"]["index_path"], "index.faiss")
        reranker_cfg = config.get("reranker", {})
        reranker_model = reranker_cfg.get("model_name") if reranker_cfg.get("enabled", True) else None
        retriever = create_retriever(
            index_path=index_path,
            docs_path=config["retrieval"]["chunks_path"],
            bm25_path=config["retrieval"]["bm25_path"],
            embedding_model=config["embedding"]["model_name"],
            reranker_model=reranker_model,
            top_k=config["retrieval"]["top_k"],
            dense_top_n=config["retrieval"]["dense_top_n"],
            bm25_top_n=config["retrieval"]["bm25_top_n"],
            rerank_top_n=config["retrieval"]["rerank_top_n"],
            rrf_k=config["retrieval"]["rrf_k"],
            aggregation=config["retrieval"].get("aggregation", "first_seen"),
            count_top_n=config["retrieval"].get("count_top_n", 20),
        )

    dynamic_pool = None
    if args.dynamic_few_shot:
        from src.agent.dynamic_few_shot import build_pool_from_config
        print(f"[dynamic-few-shot] building train pool (k={args.num_few_shot}) ...")
        dynamic_pool = build_pool_from_config(config)

    graph = None
    if not args.baseline:
        use_verify = not args.no_verify
        if args.oracle:
            print(f"[oracle mode] skipping retrieval — using gold docs (verify={use_verify})")
            graph = build_oracle_graph(llm, use_verify=use_verify, dynamic_pool=dynamic_pool, num_few_shot=args.num_few_shot)
        else:
            print(f"[langgraph] retrieval mode (verify={use_verify})")
            graph = build_graph(retriever, llm, use_verify=use_verify, dynamic_pool=dynamic_pool, num_few_shot=args.num_few_shot)
    else:
        print("[baseline] vanilla RAG, no LangGraph, no calculator tool")

    data_path = os.path.join(config["data"]["raw_dir"], f"{args.split}.json")
    run_evaluation(
        graph=graph,
        data_path=data_path,
        output_path=args.output,
        max_examples=args.max_examples,
        oracle=args.oracle,
        n_samples=args.n_samples,
        baseline=args.baseline,
        llm=llm,
        retriever=retriever,
        dynamic_pool=dynamic_pool,
    )


if __name__ == "__main__":
    main()
