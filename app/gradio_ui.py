"""Gradio web interface for the FinQA chatbot.

Three tabs:
    💬 Chat            — curated + full-dev dropdowns, oracle toggle,
                         reasoning trace, feedback.
    📊 Monitoring      — aggregate run metrics from the trace log.
    📈 Canary / Drift  — history + regression alerts.
"""

import json
import os
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from statistics import fmean
from typing import Optional

import gradio as gr

from src.observability import Tracer, set_tracer, DEFAULT_TRACE_PATH
from src.observability.drift import drift_report, DEFAULT_HISTORY_PATH


FEEDBACK_LOG_PATH = os.environ.get("FINQA_FEEDBACK_LOG", "results/feedback.jsonl")


# ── curated demo cases (top of dropdown, flagged ⭐) ───────────────

CURATED_CASES: list[dict] = [
    {
        "doc_id": "AON/2010/page_28.pdf-1",
        "question": "from the data given , how many square feet have an expiry date in 2018 ?",
        "showcases": "Direct table lookup — agent picks the right row + column, no arithmetic.",
    },
    {
        "doc_id": "MAS/2018/page_35.pdf-3",
        "question": "what was the percentage growth in the operating profit as reported between 2017 and 2018?",
        "showcases": "Two-step DSL (subtract → divide). Answer in FinQA decimal-fraction form.",
    },
    {
        "doc_id": "CB/2010/page_200.pdf-4",
        "question": "in 2010 what was the ratio of the statutory capital and surplus for combined insurance to ace tempest re usa ?",
        "showcases": "Deterministic calculator precision (4.8551); baseline LLM arithmetic would round.",
    },
    {
        "doc_id": "AAPL/2014/page_38.pdf-3",
        "question": "what was the change in amount of long term debt between 2014 and 2013?",
        "showcases": "Sign + scale sensitive. Demonstrates why retrieval accuracy matters (the decontextualized question often retrieves wrong doc — oracle toggle shows correct answer).",
    },
]


def _build_demo_choices(dev_docs: list) -> tuple[list[str], dict]:
    """Return (dropdown_choices, label_to_doc_id map).

    Layout: curated 4 first (marked ⭐), then all other dev examples sorted by doc_id.
    """
    choices: list[str] = []
    lookup: dict = {}

    curated_ids = {c["doc_id"] for c in CURATED_CASES}

    for c in CURATED_CASES:
        label = f"⭐ [{c['doc_id']}] {c['question'][:90]}"
        choices.append(label)
        lookup[label] = c["doc_id"]

    for doc in sorted(dev_docs, key=lambda d: d.doc_id):
        if doc.doc_id in curated_ids:
            continue
        q = (doc.question or "")[:90]
        label = f"[{doc.doc_id}] {q}"
        choices.append(label)
        lookup[label] = doc.doc_id

    return choices, lookup


# ── feedback + trace helpers (unchanged) ────────────────────────────

def _append_feedback(record: dict) -> None:
    os.makedirs(os.path.dirname(FEEDBACK_LOG_PATH) or ".", exist_ok=True)
    with open(FEEDBACK_LOG_PATH, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def _load_traces(path: str = DEFAULT_TRACE_PATH, limit: int = 2000) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        lines = f.readlines()[-limit:]
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _summarize_traces(events: list[dict]) -> dict:
    runs: dict = defaultdict(dict)
    for ev in events:
        rid = ev.get("run_id")
        if not rid:
            continue
        if ev["event"] == "run_start":
            runs[rid]["question"] = ev.get("question")
            runs[rid]["mode"] = ev.get("mode")
            runs[rid]["started_at"] = ev.get("ts")
            runs[rid]["nodes"] = []
        elif ev["event"] == "node_exit":
            runs[rid].setdefault("nodes", []).append({
                "node": ev.get("node"), "duration_ms": ev.get("duration_ms"),
                "error": ev.get("error"),
            })
        elif ev["event"] == "run_end":
            runs[rid]["final_answer"] = ev.get("final_answer")
            runs[rid]["elapsed_seconds"] = ev.get("elapsed_seconds")
            runs[rid]["ended_at"] = ev.get("ts")
            runs[rid]["error"] = ev.get("error")

    runs_list = [{"run_id": rid, **v} for rid, v in runs.items() if v.get("ended_at")]
    runs_list.sort(key=lambda r: r.get("ended_at", ""), reverse=True)

    latencies = [r.get("elapsed_seconds") for r in runs_list if r.get("elapsed_seconds")]
    node_counts: Counter = Counter()
    node_durations: dict = defaultdict(list)
    for r in runs_list:
        for n in r.get("nodes") or []:
            node_counts[n["node"]] += 1
            if n.get("duration_ms") is not None:
                node_durations[n["node"]].append(n["duration_ms"])

    return {
        "total_runs": len(runs_list),
        "recent": runs_list[:25],
        "avg_latency": (fmean(latencies) if latencies else None),
        "node_call_counts": dict(node_counts),
        "node_avg_duration_ms": {k: round(fmean(v), 1) for k, v in node_durations.items() if v},
    }


def _format_recent_runs_md(runs: list[dict]) -> str:
    if not runs:
        return "*No runs traced yet. Ask a question in the Chat tab.*"
    lines = [
        "| When | Mode | Question | Answer | Latency | Nodes |",
        "|---|---|---|---|---|---|",
    ]
    for r in runs[:15]:
        q = (r.get("question") or "")[:60].replace("|", "∣")
        a = str(r.get("final_answer") or "—")[:20]
        lat = f"{r['elapsed_seconds']:.1f}s" if r.get("elapsed_seconds") else "—"
        node_seq = " → ".join(n["node"] for n in (r.get("nodes") or []))
        when = (r.get("ended_at") or "")[:19].replace("T", " ")
        mode = r.get("mode") or "—"
        lines.append(f"| {when} | {mode} | {q} | {a} | {lat} | {node_seq} |")
    return "\n".join(lines)


def _format_node_stats_md(summary: dict) -> str:
    counts = summary.get("node_call_counts") or {}
    durations = summary.get("node_avg_duration_ms") or {}
    if not counts:
        return "*No node-level events yet (tracing not instrumented on nodes).*"
    lines = ["| Node | Calls | Avg duration |", "|---|---|---|"]
    for node in sorted(counts.keys()):
        d = durations.get(node)
        lines.append(f"| {node} | {counts[node]} | {f'{d:.0f} ms' if d else '—'} |")
    return "\n".join(lines)


def _format_feedback_md(limit: int = 20) -> str:
    if not os.path.exists(FEEDBACK_LOG_PATH):
        return "*No feedback recorded yet.*"
    with open(FEEDBACK_LOG_PATH) as f:
        entries = [json.loads(l) for l in f.readlines()[-limit:] if l.strip()]
    if not entries:
        return "*No feedback recorded yet.*"
    up = sum(1 for e in entries if e.get("feedback") == "up")
    down = sum(1 for e in entries if e.get("feedback") == "down")
    lines = [
        f"**Recent {len(entries)} entries** · 👍 {up} · 👎 {down}\n",
        "| Label | Question | Answer |", "|---|---|---|",
    ]
    for e in reversed(entries):
        lines.append(
            f"| {'👍' if e.get('feedback') == 'up' else '👎'} | "
            f"{(e.get('question') or '')[:60].replace('|', '∣')} | "
            f"{str(e.get('answer') or '')[:20]} |"
        )
    return "\n".join(lines)


def _format_drift_md() -> str:
    rpt = drift_report()
    history = rpt["history"]
    if not history:
        return ("*No canary history yet. Run* "
                "`python scripts/run_canary.py --notes 'baseline'` *to append the first entry.*")

    lines = [f"**Canary history:** {len(history)} runs"]
    if rpt["baseline"]:
        b = rpt["baseline"]
        lines.append(
            f"- Baseline ({b['ts'][:10]}, `{b.get('model', '—')}`, n={b.get('n', '—')}): "
            f"exec_acc={b.get('execution_accuracy', 0):.3f}, p95={b.get('latency_p95', 0):.2f}s"
        )
    if rpt["latest"] and rpt["latest"] is not rpt["baseline"]:
        lt = rpt["latest"]
        lines.append(
            f"- Latest ({lt['ts'][:10]}, `{lt.get('model', '—')}`, n={lt.get('n', '—')}): "
            f"exec_acc={lt.get('execution_accuracy', 0):.3f}, p95={lt.get('latency_p95', 0):.2f}s"
        )
    if rpt.get("message"):
        lines.append(f"\n*{rpt['message']}*")

    if rpt["alerts"]:
        lines.append("\n### 🚨 Drift alerts")
        lines.append("| Signal | Severity | Baseline | Latest | Δ |")
        lines.append("|---|---|---|---|---|")
        for a in rpt["alerts"]:
            lines.append(
                f"| {a['signal']} | {a['severity']} | "
                f"{a['baseline']:.4f} | {a['latest']:.4f} | {a['delta']:+.4f} |"
            )
    elif len(history) > 1:
        lines.append("\n✅ No drift detected.")

    if len(history) > 1:
        lines.append("\n### Run history")
        lines.append("| When | Model | n | exec_acc | p95 | Notes |")
        lines.append("|---|---|---|---|---|---|")
        for h in history[-15:]:
            lines.append(
                f"| {h['ts'][:19].replace('T',' ')} | "
                f"{str(h.get('model', '—')).split('/')[-1]} | "
                f"{h.get('n', '—')} | {h.get('execution_accuracy', 0):.3f} | "
                f"{h.get('latency_p95', 0):.2f}s | "
                f"{(h.get('notes') or '')[:40]} |"
            )
    return "\n".join(lines)


# ── UI construction ─────────────────────────────────────────────────

def create_ui(
    retrieval_graph,
    oracle_graph,
    retriever=None,
    dev_docs: Optional[list] = None,
    meta: Optional[dict] = None,
):
    """Create the Gradio Blocks interface.

    Args:
        retrieval_graph: Compiled LangGraph with RETRIEVE (end-to-end).
        oracle_graph:    Compiled LangGraph with INJECT_ORACLE (gold-doc mode).
        retriever:       Passed through to the monitoring tab if needed.
        dev_docs:        Full dev FinQADocument list for the dropdown.
        meta:            Dict with model_name, use_verify, etc. for status badges.
    """
    from src.agent.graph import run_question

    dev_docs = dev_docs or []
    meta = meta or {}
    tracer = Tracer(DEFAULT_TRACE_PATH)
    set_tracer(tracer)

    # Build dropdown + doc lookup
    demo_choices, demo_label_to_doc_id = _build_demo_choices(dev_docs)
    doc_by_id = {d.doc_id: d for d in dev_docs}

    # (Stateless design: all per-request context comes from Dropdown value
    # and Chatbot history. Avoids gr.State ↔ Gradio 5.x StateHolder bugs.)

    # ── handlers ────────────────────────────────────────────────────

    def on_demo_select(label):
        try:
            if not label:
                return "", ""
            doc_id = demo_label_to_doc_id.get(label)
            if not doc_id or doc_id not in doc_by_id:
                return "", "*(doc not found in dev set)*"
            doc = doc_by_id[doc_id]
            showcase = next(
                (c["showcases"] for c in CURATED_CASES if c["doc_id"] == doc_id),
                None,
            )
            lines = [
                f"**Doc ID**: `{doc_id}`",
                f"**Expected gold answer**: `{doc.gold_answer}`",
                f"**Gold program**: `{doc.gold_program or '(n/a)'}`",
            ]
            if showcase:
                lines.append(f"**Showcases**: {showcase}")
            return str(doc.question or ""), "  \n".join(lines)
        except Exception as e:
            return "", f"*Error loading demo: {e}*"

    def process_question(question, history, dropdown_label, oracle_toggle):
        # Normalize inputs
        question = (question or "").strip()
        history = history or []
        if not question:
            return history, "_(enter a question)_", "", ""

        # Look up doc from current dropdown value (stateless)
        selected_doc_id = demo_label_to_doc_id.get(dropdown_label) if dropdown_label else None
        selected_gold = None
        if selected_doc_id and selected_doc_id in doc_by_id:
            selected_gold = doc_by_id[selected_doc_id].gold_answer

        run_id = uuid.uuid4().hex[:12]
        started = datetime.now(timezone.utc)

        use_oracle = bool(oracle_toggle and selected_doc_id and selected_doc_id in doc_by_id)
        mode = "oracle" if use_oracle else "retrieval"
        try:
            tracer.run_start(run_id, question=question, mode=mode)
        except Exception:
            pass

        oracle_doc = None
        graph = retrieval_graph
        if use_oracle:
            try:
                doc = doc_by_id[selected_doc_id]
                oracle_doc = {
                    "content": doc.get_context_for_llm(),
                    "doc_id": doc.doc_id, "table": doc.table, "table_md": doc.table_md,
                    "pre_text": doc.pre_text, "post_text": doc.post_text,
                }
                graph = oracle_graph
            except Exception:
                mode = "retrieval (oracle setup failed)"
                oracle_doc = None
                graph = retrieval_graph

        answer = "(no answer)"
        error_msg = ""
        result: dict = {}
        try:
            result = run_question(graph, question, oracle_doc=oracle_doc)
            answer = result.get("final_answer") or "(no answer)"
            error_msg = result.get("error") or ""
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            answer = f"(error) {error_msg[:80]}"
            result = {"retrieved_docs": [], "reasoning": None}

        elapsed = (datetime.now(timezone.utc) - started).total_seconds()
        try:
            tracer.run_end(
                run_id, final_answer=str(answer),
                elapsed_seconds=elapsed, error=error_msg or None,
            )
        except Exception:
            pass

        # Retrieved docs panel
        docs_display_parts: list[str] = []
        for i, doc in enumerate((result.get("retrieved_docs") or [])[:3]):
            md = doc.get("metadata", {}) if isinstance(doc, dict) else {}
            did = md.get("doc_id", "unknown")
            score = md.get("score", 0) or 0
            content = (doc.get("content") or "") if isinstance(doc, dict) else ""
            if len(content) > 600:
                content = content[:600] + "…"
            docs_display_parts.append(
                f"**Doc {i+1}** — `{did}` (score {float(score):.3f})\n\n{content}"
            )
        docs_display = "\n\n---\n\n".join(docs_display_parts) or "_No documents retrieved._"

        # Reasoning trace panel
        rtext_parts = [f"**Mode**: `{mode}`"]
        if use_oracle and selected_doc_id:
            rtext_parts[0] += f" (gold doc: `{selected_doc_id}`)"
        reasoning = result.get("reasoning") or {}
        if isinstance(reasoning, dict) and reasoning:
            rtext_parts.append(f"**Action**: `{reasoning.get('action', '—')}`")
            if reasoning.get("reasoning"):
                rtext_parts.append(f"**Reasoning**: {reasoning['reasoning']}")
            if reasoning.get("expression"):
                rtext_parts.append(f"**Expression**: `{reasoning['expression']}`")
        if result.get("predicted_program"):
            rtext_parts.append(f"**DSL program**: `{result['predicted_program']}`")
        if result.get("calc_result"):
            rtext_parts.append(f"**Calculator**:\n```\n{result['calc_result']}\n```")
        if error_msg:
            rtext_parts.append(f"**Error**: {error_msg}")

        # Verdict vs gold
        if selected_gold is not None:
            try:
                from src.evaluation.metrics import execution_accuracy
                ok = execution_accuracy(str(answer), float(selected_gold))
                badge = "✅ matches gold" if ok else "❌ differs from gold"
                rtext_parts.append(f"**vs gold** ({selected_gold}): {badge}")
            except Exception:
                pass
        rtext_parts.append(f"*Latency: {elapsed:.2f}s · run_id: `{run_id}`*")
        rtext = "\n\n".join(rtext_parts)

        # Chatbot in "messages" mode: list of {role, content} dicts
        new_history = list(history) + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": str(answer)},
        ]
        # Log latest Q/A for feedback button to pick up (no gr.State needed)
        _append_feedback({
            "run_id": run_id, "question": question, "answer": str(answer),
            "mode": mode, "feedback": None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        return new_history, docs_display, rtext, ""

    def record_feedback(label):
        # Read the most recent feedback-log entry and append a feedback label.
        # Simpler than carrying state across callbacks; good enough for a demo.
        try:
            if not os.path.exists(FEEDBACK_LOG_PATH):
                return "No question to rate yet."
            with open(FEEDBACK_LOG_PATH) as f:
                lines = [l for l in f.readlines() if l.strip()]
            if not lines:
                return "No question to rate yet."
            last = json.loads(lines[-1])
            last["feedback"] = label
            with open(FEEDBACK_LOG_PATH, "a") as f:
                f.write(json.dumps(last, default=str) + "\n")
            return f"Thanks — recorded feedback ({'👍' if label == 'up' else '👎'})."
        except Exception as e:
            return f"(feedback save failed: {e})"

    def refresh_monitoring():
        events = _load_traces()
        summary = _summarize_traces(events)
        avg_lat = summary.get("avg_latency")
        avg_lat_str = f"{avg_lat:.2f}s" if avg_lat else "—"
        overview = (
            f"**Total traced runs:** {summary['total_runs']}  \n"
            f"**Avg latency:** {avg_lat_str}"
        )
        return (
            overview,
            _format_recent_runs_md(summary.get("recent") or []),
            _format_node_stats_md(summary),
            _format_feedback_md(),
        )

    def refresh_drift():
        return _format_drift_md()

    # ── layout ──────────────────────────────────────────────────────

    with gr.Blocks(title="FinQA Chatbot", theme=gr.themes.Soft()) as app:
        # Status header — always visible across tabs
        gr.Markdown(
            "# FinQA Chatbot\n"
            "Agentic-RAG with a deterministic calculator. "
            "LangGraph orchestration + vLLM-served Qwen2.5-7B."
        )
        status_badges = (
            f"**🤖 Model**: `{meta.get('model_name', '—')}` ·  "
            f"**🧠 Mode**: LangGraph + Tool ·  "
            f"**⚡ VERIFY**: {'on' if meta.get('use_verify') else 'off (production)'} ·  "
            f"**🔍 Retriever top-K**: {meta.get('retriever_top_k', '—')} ·  "
            f"**📚 Dev examples loaded**: {len(dev_docs)}"
        )
        gr.Markdown(status_badges)

        with gr.Tabs():
            # ─── 💬 Chat tab ──────────────────────────────────────
            with gr.Tab("💬 Chat"):
                with gr.Row():
                    with gr.Column(scale=2):
                        chatbot = gr.Chatbot(label="Conversation", height=380, type="messages")

                        with gr.Accordion("Curated demos + full dev (click to load)", open=True):
                            demo_dropdown = gr.Dropdown(
                                choices=demo_choices,
                                label=f"Select example — {len(demo_choices)} available (⭐ = curated)",
                                value=None,
                                interactive=True,
                                filterable=True,
                                allow_custom_value=False,
                            )
                            demo_info = gr.Markdown()
                            oracle_toggle = gr.Checkbox(
                                label="🔮 Oracle mode for selected demo (inject gold doc, skip retrieval)",
                                value=True,
                                info=(
                                    "When on AND a demo is selected, the gold page is fed "
                                    "directly to the agent (isolates reasoning). "
                                    "Turn off to see end-to-end behaviour including retrieval."
                                ),
                            )

                        with gr.Row():
                            question_input = gr.Textbox(
                                label="Your question",
                                placeholder="e.g., What was the percentage change in revenue from 2014 to 2015?",
                                scale=4,
                            )
                            submit_btn = gr.Button("Ask", variant="primary", scale=1)

                        with gr.Row():
                            up_btn = gr.Button("👍 Correct", scale=1)
                            down_btn = gr.Button("👎 Incorrect", scale=1)
                            clear_btn = gr.Button("🗑️ Clear conversation", scale=1)
                            feedback_status = gr.Markdown("")

                        gr.Markdown(
                            "> **Note**: each Ask is an independent RAG call — the system does "
                            "*not* use previous turns as context. Follow-up questions like "
                            "*\"How did that number come?\"* will re-retrieve from scratch and "
                            "likely get a different, unrelated document. To see the derivation "
                            "for an answer, check the **Reasoning trace** panel on the right "
                            "(it shows the DSL program the calculator ran). "
                            "Click 🗑️ between questions if you want a clean slate."
                        )

                    with gr.Column(scale=1):
                        gr.Markdown("### Retrieved documents")
                        docs_output = gr.Markdown()
                        gr.Markdown("---")
                        gr.Markdown("### Reasoning trace")
                        reasoning_output = gr.Markdown()

                gr.Markdown(
                    "**Tips**\n"
                    "- Percentages are returned as decimal fractions (e.g. `0.1434` for 14.34%) — FinQA convention.\n"
                    "- The DSL program in the trace shows **exactly** what the calculator computed; every answer is auditable.\n"
                    "- Oracle mode lets you see reasoning quality in isolation; retrieval mode shows end-to-end behaviour."
                )

                def on_demo_select_with_clear(label):
                    """Also clears the chatbot so demo answers don't mix with previous turns."""
                    q, info = on_demo_select(label)
                    return q, info, [], "_No documents retrieved._", ""

                demo_dropdown.change(
                    on_demo_select_with_clear,
                    inputs=[demo_dropdown],
                    outputs=[question_input, demo_info, chatbot, docs_output, reasoning_output],
                )
                submit_btn.click(
                    process_question,
                    inputs=[question_input, chatbot, demo_dropdown, oracle_toggle],
                    outputs=[chatbot, docs_output, reasoning_output, question_input],
                )
                question_input.submit(
                    process_question,
                    inputs=[question_input, chatbot, demo_dropdown, oracle_toggle],
                    outputs=[chatbot, docs_output, reasoning_output, question_input],
                )
                up_btn.click(lambda: record_feedback("up"), outputs=[feedback_status])
                down_btn.click(lambda: record_feedback("down"), outputs=[feedback_status])
                clear_btn.click(
                    lambda: ([], "_No documents retrieved._", "", ""),
                    outputs=[chatbot, docs_output, reasoning_output, feedback_status],
                )

            # ─── 📊 Monitoring tab ────────────────────────────────
            with gr.Tab("📊 Monitoring"):
                gr.Markdown(
                    "### Aggregate metrics from the trace log\n"
                    f"Every Chat question and batch eval appends events to "
                    f"`{DEFAULT_TRACE_PATH}`. Click refresh after new activity."
                )
                refresh_btn = gr.Button("🔄 Refresh", variant="secondary")
                overview_md = gr.Markdown()
                gr.Markdown("#### Recent runs")
                recent_md = gr.Markdown()
                gr.Markdown("#### Node statistics")
                node_md = gr.Markdown()
                gr.Markdown("#### User feedback")
                feedback_md = gr.Markdown()
                refresh_btn.click(
                    refresh_monitoring,
                    outputs=[overview_md, recent_md, node_md, feedback_md],
                )
                app.load(refresh_monitoring,
                         outputs=[overview_md, recent_md, node_md, feedback_md])

            # ─── 📈 Canary / Drift tab ────────────────────────────
            with gr.Tab("📈 Canary / Drift"):
                gr.Markdown(
                    "### Canary accuracy over time\n"
                    f"Run `python scripts/run_canary.py --n 50 --notes '...'` to "
                    f"append a new entry to `{DEFAULT_HISTORY_PATH}`. "
                    "Alerts fire when execution accuracy drops ≥ 3 pp, parse success drops ≥ 5 pp, "
                    "or p95 latency grows ≥ 1.5×."
                )
                drift_refresh_btn = gr.Button("🔄 Refresh", variant="secondary")
                drift_md = gr.Markdown()
                drift_refresh_btn.click(refresh_drift, outputs=[drift_md])
                app.load(refresh_drift, outputs=[drift_md])

    return app
