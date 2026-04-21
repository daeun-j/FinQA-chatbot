"""Batch evaluation runner for FinQA."""

import json
import os
import time
from typing import Optional

from src.data_processing.loader import load_finqa_file
from src.evaluation.metrics import batch_evaluate, execution_accuracy
from src.agent.self_consistency import vote_on_answers
from src.agent.baseline import run_baseline_question


def run_evaluation(
    graph,
    data_path: str,
    output_path: str = "results/eval_results.json",
    max_examples: Optional[int] = None,
    verbose: bool = True,
    oracle: bool = False,
    n_samples: int = 1,
    baseline: bool = False,
    llm=None,
    retriever=None,
    dynamic_pool=None,
) -> dict:
    """Run the agent over a dataset split and evaluate.

    Args:
        graph: Compiled LangGraph agent.
        data_path: Path to dev.json or test.json.
        output_path: Where to save results JSON.
        max_examples: Limit number of examples (for debugging).
        verbose: Print progress.

    Returns:
        Evaluation results dict.
    """
    documents = load_finqa_file(data_path)
    if max_examples:
        documents = documents[:max_examples]

    predictions = []
    errors = []

    for idx, doc in enumerate(documents):
        if verbose:
            print(f"[{idx+1}/{len(documents)}] {doc.doc_id}: {doc.question[:60]}...")

        start_time = time.time()

        try:
            from src.agent.graph import run_question
            oracle_doc = None
            if oracle:
                oracle_doc = {
                    "content": doc.get_context_for_llm(),
                    "doc_id": doc.doc_id,
                    "table": doc.table,
                    "table_md": doc.table_md,
                    "pre_text": doc.pre_text,
                    "post_text": doc.post_text,
                }
            if baseline:
                if n_samples > 1:
                    samples = [
                        run_baseline_question(llm, retriever, doc.question, oracle_doc=oracle_doc, dynamic_pool=dynamic_pool)
                        for _ in range(n_samples)
                    ]
                    result = vote_on_answers(samples)
                else:
                    result = run_baseline_question(llm, retriever, doc.question, oracle_doc=oracle_doc, dynamic_pool=dynamic_pool)
            elif n_samples > 1:
                samples = [
                    run_question(graph, doc.question, oracle_doc=oracle_doc)
                    for _ in range(n_samples)
                ]
                result = vote_on_answers(samples)
            else:
                result = run_question(graph, doc.question, oracle_doc=oracle_doc)

            predicted_answer = result.get("final_answer", "")
            predicted_program = result.get("predicted_program") or ""
            elapsed = time.time() - start_time

            pred = {
                "doc_id": doc.doc_id,
                "question": doc.question,
                "predicted_answer": predicted_answer,
                "predicted_program": predicted_program,
                "gold_answer": doc.gold_answer,
                "gold_program": doc.gold_program,
                "elapsed_seconds": elapsed,
            }
            predictions.append(pred)

            if verbose:
                if doc.gold_answer is None:
                    exec_match = "NO_GOLD"
                else:
                    exec_match = "OK" if execution_accuracy(predicted_answer, doc.gold_answer) else "MISS"
                prog_match = "OK" if predicted_program and doc.gold_program and \
                    predicted_program.replace(" ", "").lower() == doc.gold_program.replace(" ", "").lower() else "MISS"
                vote_str = ""
                if n_samples > 1 and "vote_count" in result:
                    vote_str = f" vote={result['vote_count']}/{result['n_samples']}"
                print(f"  -> exec[{exec_match}] prog[{prog_match}] pred={predicted_answer}, gold={doc.gold_answer}{vote_str} ({elapsed:.1f}s)")

        except Exception as e:
            errors.append({"doc_id": doc.doc_id, "error": str(e)})
            predictions.append({
                "doc_id": doc.doc_id,
                "question": doc.question,
                "predicted_answer": "",
                "gold_answer": doc.gold_answer,
                "gold_program": doc.gold_program,
                "elapsed_seconds": time.time() - start_time,
            })
            if verbose:
                print(f"  -> ERROR: {e}")

    # Evaluate
    eval_results = batch_evaluate(predictions)
    eval_results["errors"] = errors

    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(eval_results, f, indent=2, default=str)

    if verbose:
        print(f"\n=== Evaluation Results ===")
        print(f"Execution Accuracy: {eval_results['execution_accuracy']:.4f} ({eval_results['exec_correct']}/{eval_results['total']})")
        print(f"Program Accuracy:   {eval_results['program_accuracy']:.4f}")
        print(f"Errors:             {len(errors)}")
        print(f"Results saved to:   {output_path}")

    return eval_results
