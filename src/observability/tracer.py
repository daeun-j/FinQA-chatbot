"""Minimal JSONL event tracer for the agent.

Writes `run_start`, `node_enter`, `node_exit`, `run_end` events to a file.
The Gradio Monitoring tab reads these to show recent runs + node statistics.
Swappable for LangSmith in production — the call-site contract is just
`start_run()`, `node_enter()`, `node_exit()`, `end_run()`.
"""

import json
import os
from datetime import datetime, timezone
from typing import Optional


DEFAULT_TRACE_PATH = os.environ.get("FINQA_TRACE_PATH", "results/traces.jsonl")


class Tracer:
    def __init__(self, path: str = DEFAULT_TRACE_PATH):
        self.path = path
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    def _write(self, event: dict) -> None:
        event.setdefault("ts", datetime.now(timezone.utc).isoformat())
        try:
            with open(self.path, "a") as f:
                f.write(json.dumps(event, default=str) + "\n")
        except Exception:
            # Never let tracing errors crash the agent.
            pass

    # Convenience event emitters
    def run_start(self, run_id: str, question: str, mode: str = "chat") -> None:
        self._write({
            "event": "run_start", "run_id": run_id,
            "question": question, "mode": mode,
        })

    def node_enter(self, run_id: str, node: str) -> None:
        self._write({"event": "node_enter", "run_id": run_id, "node": node})

    def node_exit(
        self,
        run_id: str,
        node: str,
        duration_ms: Optional[float] = None,
        error: Optional[str] = None,
    ) -> None:
        self._write({
            "event": "node_exit", "run_id": run_id, "node": node,
            "duration_ms": duration_ms, "error": error,
        })

    def run_end(
        self,
        run_id: str,
        final_answer: Optional[str] = None,
        elapsed_seconds: Optional[float] = None,
        error: Optional[str] = None,
    ) -> None:
        self._write({
            "event": "run_end", "run_id": run_id,
            "final_answer": final_answer,
            "elapsed_seconds": elapsed_seconds,
            "error": error,
        })


_GLOBAL_TRACER: Optional[Tracer] = None


def set_tracer(tracer: Tracer) -> None:
    global _GLOBAL_TRACER
    _GLOBAL_TRACER = tracer


def get_tracer() -> Optional[Tracer]:
    return _GLOBAL_TRACER
