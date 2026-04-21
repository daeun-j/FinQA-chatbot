"""Drift detection across canary eval runs.

A canary history is a JSONL file of eval-run summaries, each at minimum:
    {
      "ts": "2026-04-21T14:00:00+00:00",
      "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
      "n": 100,
      "execution_accuracy": 0.60,
      "program_accuracy": 0.43,
      "parse_success": 0.97,
      "latency_p95": 8.5,
      "notes": "post-prompt-tightening"
    }

`drift_report()` returns the history + alert list. Thresholds (hard-coded
here, tunable per deployment):
  • execution_accuracy drop ≥ 3 pp     → critical
  • parse_success drop ≥ 5 pp          → warning
  • latency_p95 growth ≥ 1.5×          → warning
"""

import json
import os
from typing import Optional


DEFAULT_HISTORY_PATH = os.environ.get("FINQA_CANARY_HISTORY", "results/canary_history.jsonl")


def _load_history(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def drift_report(path: str = DEFAULT_HISTORY_PATH) -> dict:
    history = _load_history(path)
    result = {
        "history": history,
        "baseline": history[0] if history else None,
        "latest": history[-1] if history else None,
        "alerts": [],
        "message": "",
    }

    if not history:
        result["message"] = "No canary history yet."
        return result

    if len(history) == 1:
        result["message"] = "Only baseline recorded. Run another canary to enable drift detection."
        return result

    baseline = result["baseline"]
    latest = result["latest"]
    alerts: list[dict] = []

    # execution_accuracy drop ≥ 3pp
    b_exec = baseline.get("execution_accuracy")
    l_exec = latest.get("execution_accuracy")
    if b_exec is not None and l_exec is not None:
        delta = l_exec - b_exec
        if delta <= -0.03:
            alerts.append({
                "signal": "execution_accuracy", "severity": "critical",
                "baseline": b_exec, "latest": l_exec, "delta": delta,
            })

    # parse_success drop ≥ 5pp
    b_ps = baseline.get("parse_success", 1.0)
    l_ps = latest.get("parse_success", 1.0)
    if (l_ps - b_ps) <= -0.05:
        alerts.append({
            "signal": "parse_success", "severity": "warning",
            "baseline": b_ps, "latest": l_ps, "delta": (l_ps - b_ps),
        })

    # latency_p95 growth ≥ 1.5×
    b_p95 = baseline.get("latency_p95")
    l_p95 = latest.get("latency_p95")
    if b_p95 and l_p95 and l_p95 >= 1.5 * b_p95:
        alerts.append({
            "signal": "latency_p95", "severity": "warning",
            "baseline": b_p95, "latest": l_p95, "delta": l_p95 / max(b_p95, 1e-6),
        })

    result["alerts"] = alerts
    return result
