"""Canary evaluation runner — writes a one-line summary to canary_history.jsonl.

Run after any config/prompt/model change to append a new canary entry.
The Gradio Canary/Drift tab reads this file and alerts on regressions.

This runner is intentionally verbose (progress line per question, loud errors)
so regressions don't hide behind silent exception swallowing.
"""

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.graph import build_graph, build_oracle_graph, run_question
from src.data_processing.loader import load_finqa_file
from src.evaluation.metrics import execution_accuracy
from src.llm.vllm_client import create_llm
from src.observability.drift import DEFAULT_HISTORY_PATH


def _check_vllm(base_url: str, model: str) -> None:
    """Fail fast if vLLM isn't serving the expected model."""
    import requests
    try:
        r = requests.get(f"{base_url.rstrip('/')}/models", timeout=5)
        r.raise_for_status()
        data = r.json()
        served = [m.get("id") for m in data.get("data", [])]
        if model not in served:
            raise RuntimeError(
                f"vLLM is running but serves {served}, not '{model}'. "
                f"Update configs/config.yaml model.name to match, or restart vLLM."
            )
        print(f"[canary] vLLM OK — serving '{model}'")
    except Exception as e:
        raise RuntimeError(
            f"Cannot reach vLLM at {base_url}. Is the server running?\n"
            f"Original error: {e}"
        ) from e


def main():
    ap = argparse.ArgumentParser(description="Run FinQA canary and append to history.")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--n", type=int, default=50, help="Canary size (default 50).")
    ap.add_argument("--no-oracle", action="store_true",
                    help="Run end-to-end with retrieval instead of oracle mode.")
    ap.add_argument("--notes", default="", help="Free-form notes for this run.")
    ap.add_argument("--history-path", default=DEFAULT_HISTORY_PATH)
    ap.add_argument("--verbose", action="store_true", default=True)
    args = ap.parse_args()
    oracle_mode = not args.no_oracle

    # ── Config + connectivity checks ───────────────────────────────
    if not os.path.exists(args.config):
        print(f"ERROR: config not found: {args.config}", file=sys.stderr)
        sys.exit(1)
    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_path = os.path.join(config["data"]["raw_dir"], f"{args.split}.json")
    if not os.path.exists(data_path):
        print(f"ERROR: data file not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    _check_vllm(config["model"]["vllm_base_url"], config["model"]["name"])

    # ── Build graph ────────────────────────────────────────────────
    print(f"[canary] mode: {'oracle' if oracle_mode else 'retrieval'}, n={args.n}")
    llm = create_llm(
        base_url=config["model"]["vllm_base_url"],
        model_name=config["model"]["name"],
        temperature=config["model"]["temperature"],
        max_tokens=config["model"]["max_tokens"],
    )

    if oracle_mode:
        graph = build_oracle_graph(llm, use_verify=False)
        retriever = None
    else:
        # Retrieval mode: need a live retriever
        retrieval_cfg = config["retrieval"]
        index_path = os.path.join(retrieval_cfg["index_path"], "index.faiss")
        required = [index_path, retrieval_cfg["chunks_path"], retrieval_cfg["bm25_path"]]
        missing = [p for p in required if not os.path.exists(p)]
        if missing:
            print(f"ERROR: index files missing: {missing}", file=sys.stderr)
            sys.exit(1)
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
        )
        graph = build_graph(retriever, llm, use_verify=False)

    # ── Load examples ──────────────────────────────────────────────
    examples = load_finqa_file(data_path)[: args.n]
    if not examples:
        print(f"ERROR: no examples loaded from {data_path}", file=sys.stderr)
        sys.exit(1)
    print(f"[canary] loaded {len(examples)} examples from {data_path}")

    # ── Run ────────────────────────────────────────────────────────
    correct = 0
    parse_success = 0
    errors = 0
    latencies: list[float] = []
    started_at = time.time()

    for i, doc in enumerate(examples, start=1):
        oracle_doc = None
        if oracle_mode:
            oracle_doc = {
                "content": doc.get_context_for_llm(),
                "doc_id": doc.doc_id, "table": doc.table, "table_md": doc.table_md,
                "pre_text": doc.pre_text, "post_text": doc.post_text,
            }
        t0 = time.time()
        answer, err = "", None
        try:
            result = run_question(graph, doc.question, oracle_doc=oracle_doc)
            answer = result.get("final_answer", "") or ""
            if execution_accuracy(answer, doc.gold_answer):
                correct += 1
            if result.get("reasoning"):
                parse_success += 1
        except Exception as e:
            errors += 1
            err = str(e)
            traceback.print_exc(file=sys.stderr)
        dt = time.time() - t0
        latencies.append(dt)

        if args.verbose:
            status = "ERR" if err else ("OK" if execution_accuracy(answer, doc.gold_answer) else "MISS")
            print(f"  [{i:>3}/{len(examples)}] {doc.doc_id:<30} "
                  f"pred={str(answer)[:20]:<20} gold={doc.gold_answer} [{status}] ({dt:.1f}s)")

    # ── Summary ────────────────────────────────────────────────────
    n = len(examples)
    latencies.sort()
    p95 = latencies[min(int(0.95 * n), n - 1)] if latencies else 0.0
    p50 = latencies[n // 2] if latencies else 0.0
    total_time = time.time() - started_at

    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "model": config["model"]["name"],
        "n": n,
        "execution_accuracy": correct / n if n else 0.0,
        "parse_success": parse_success / n if n else 0.0,
        "latency_p50": p50,
        "latency_p95": p95,
        "errors": errors,
        "mode": "oracle" if oracle_mode else "retrieval",
        "notes": args.notes,
    }

    os.makedirs(os.path.dirname(args.history_path) or ".", exist_ok=True)
    with open(args.history_path, "a") as f:
        f.write(json.dumps(entry) + "\n")

    print("\n" + "=" * 60)
    print(f"[canary] appended to {args.history_path}")
    print(f"  execution_accuracy: {entry['execution_accuracy']:.4f}  ({correct}/{n})")
    print(f"  parse_success:      {entry['parse_success']:.4f}")
    print(f"  latency p50/p95:    {p50:.2f}s / {p95:.2f}s")
    print(f"  errors:             {errors}")
    print(f"  total time:         {total_time:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
