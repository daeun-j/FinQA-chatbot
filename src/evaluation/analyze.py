"""Error analysis for FinQA evaluation results."""

import json
from collections import Counter


def analyze_errors(results_path: str) -> dict:
    """Analyze evaluation results to categorize failure types.

    Categories:
    - retrieval_failure: correct doc not in top-k (requires gold_doc_id check)
    - parse_failure: LLM output couldn't be parsed as JSON
    - calculation_error: wrong calculation formula
    - extraction_error: wrong numbers extracted from document
    - runtime_error: exception during execution

    Args:
        results_path: Path to eval_results.json.

    Returns:
        Analysis summary dict.
    """
    with open(results_path, "r") as f:
        results = json.load(f)

    details = results.get("details", [])
    errors = results.get("errors", [])

    # Basic stats
    correct = sum(1 for d in details if d.get("exec_correct"))
    incorrect = [d for d in details if not d.get("exec_correct")]

    # Categorize incorrect predictions
    categories = Counter()
    for pred in incorrect:
        predicted = str(pred.get("predicted", ""))
        gold = pred.get("gold", 0)

        if not predicted or predicted == "Unable to determine answer":
            categories["parse_failure"] += 1
        elif predicted == "0" and gold != 0:
            categories["extraction_error"] += 1
        else:
            categories["calculation_error"] += 1

    for err in errors:
        categories["runtime_error"] += 1

    analysis = {
        "total": len(details),
        "correct": correct,
        "incorrect": len(incorrect),
        "accuracy": correct / len(details) if details else 0,
        "error_categories": dict(categories),
        "runtime_errors": len(errors),
    }

    # Print summary
    print("=== Error Analysis ===")
    print(f"Total: {analysis['total']}")
    print(f"Correct: {analysis['correct']} ({analysis['accuracy']:.2%})")
    print(f"Incorrect: {analysis['incorrect']}")
    print(f"\nError Categories:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

    return analysis


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "results/eval_results.json"
    analyze_errors(path)
