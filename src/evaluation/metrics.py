"""Evaluation metrics for FinQA."""

import re
from typing import Optional


def execution_accuracy(predicted: str, gold: float, tolerance: float = 0.01) -> bool:
    """Check if predicted answer matches gold within relative tolerance.

    Accepts the prediction in either the FinQA "fraction" form (0.14 for 14%)
    or the "percentage" form (14.0 for 14%): FinQA's gold labels mix the two
    inconsistently, so most reimplementations treat `pred == gold * 100` and
    `pred == gold / 100` as equivalent. Keeps the `gold == 0` edge case.
    """
    if gold is None:
        return False

    try:
        pred_clean = str(predicted).replace(",", "").replace("%", "").strip()
        pred_val = float(pred_clean)
    except (ValueError, TypeError):
        return False

    if gold == 0:
        return abs(pred_val) < 1e-6

    def close(a: float, b: float) -> bool:
        return abs(a - b) / max(abs(b), 1e-6) < tolerance

    return close(pred_val, gold) or close(pred_val, gold * 100) or close(pred_val, gold / 100)


def program_accuracy(predicted_program: str, gold_program: str) -> bool:
    """Check exact match of predicted vs gold program (after normalization).

    Args:
        predicted_program: Predicted DSL program string.
        gold_program: Gold DSL program string.

    Returns:
        True if programs match after whitespace normalization.
    """
    def normalize(prog: str) -> str:
        # Remove all whitespace, lowercase
        return re.sub(r"\s+", "", prog.lower())

    return normalize(predicted_program) == normalize(gold_program)


def batch_evaluate(
    predictions: list[dict],
) -> dict:
    """Evaluate a batch of predictions.

    Args:
        predictions: List of dicts with keys:
            - predicted_answer: str
            - gold_answer: float
            - predicted_program: str (optional)
            - gold_program: str (optional)
            - doc_id: str

    Returns:
        Dict with execution_accuracy, program_accuracy, and per-example results.
    """
    exec_correct = 0
    prog_correct = 0
    total = len(predictions)
    details = []

    for pred in predictions:
        exec_match = execution_accuracy(
            pred["predicted_answer"],
            pred["gold_answer"],
        )
        prog_match = False
        if pred.get("predicted_program") and pred.get("gold_program"):
            prog_match = program_accuracy(
                pred["predicted_program"],
                pred["gold_program"],
            )

        if exec_match:
            exec_correct += 1
        if prog_match:
            prog_correct += 1

        details.append({
            "doc_id": pred.get("doc_id", ""),
            "exec_correct": exec_match,
            "prog_correct": prog_match,
            "predicted": pred["predicted_answer"],
            "gold": pred["gold_answer"],
        })

    return {
        "execution_accuracy": exec_correct / total if total > 0 else 0.0,
        "program_accuracy": prog_correct / total if total > 0 else 0.0,
        "total": total,
        "exec_correct": exec_correct,
        "prog_correct": prog_correct,
        "details": details,
    }
