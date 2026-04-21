"""Self-consistency voting over multiple agent samples.

Calls the agent N times at high temperature, then votes on the numeric answer.
Two answers count as the same vote if they're equal under FinQA's percentage
ambiguity (x ≡ x*100 ≡ x/100 within tolerance), so we don't split the vote
just because samples disagree on output formatting.
"""

from collections import Counter
from typing import Optional


def _to_float(x: str) -> Optional[float]:
    try:
        return float(str(x).replace(",", "").replace("%", "").strip())
    except (ValueError, TypeError):
        return None


def _equivalent(a: float, b: float, tol: float = 0.01) -> bool:
    if b == 0:
        return abs(a) < 1e-6
    rel = lambda x, y: abs(x - y) / max(abs(y), 1e-6) < tol
    return rel(a, b) or rel(a, b * 100) or rel(a, b / 100)


def vote_on_answers(samples: list[dict]) -> dict:
    """Pick the winning sample by numeric majority vote.

    Args:
        samples: List of dicts with at least "final_answer" and
                 "predicted_program". Order should be the order they were
                 sampled in (used as a tiebreaker — earlier wins).

    Returns:
        The winning sample dict, augmented with vote-tracking fields:
            - vote_count: how many samples agreed with the winner
            - n_samples: total sample count
            - all_answers: raw answer strings from each sample
    """
    if not samples:
        return {}

    # Group samples whose answers are numerically equivalent
    groups: list[list[dict]] = []
    for s in samples:
        val = _to_float(s.get("final_answer", ""))
        placed = False
        if val is not None:
            for g in groups:
                rep = _to_float(g[0].get("final_answer", ""))
                if rep is not None and _equivalent(val, rep):
                    g.append(s)
                    placed = True
                    break
        if not placed:
            groups.append([s])

    # Largest group wins; ties broken by earliest first occurrence (stable)
    groups.sort(key=lambda g: -len(g))
    winners = groups[0]
    winner = winners[0]

    return {
        **winner,
        "vote_count": len(winners),
        "n_samples": len(samples),
        "all_answers": [s.get("final_answer", "") for s in samples],
    }
