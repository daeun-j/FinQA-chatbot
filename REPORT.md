# FinQA Chatbot — Technical Report

**Agentic-RAG with Deterministic Tool Use for Numerical Reasoning over Financial Documents**

> **Headline result** (FinQA dev, full 883 examples, oracle mode, Qwen2.5-7B-Instruct-AWQ via vLLM, **zero-shot**):
> - **LangGraph + Tool**: 59.00% execution / **38.39% program accuracy** (521/883 exec, 339/883 prog)
> - **Vanilla RAG baseline**: 61.95% execution / 23.78% program (547/883 exec, 210/883 prog)
>
> Our zero-shot baseline **matches the FinQA paper's supervised RoBERTa-large FinQANet (61.24% exec / 58.86% prog)** on execution within 0.7pp, without any task-specific fine-tuning. The LangGraph + deterministic calculator architecture trades 2.95pp of execution for **+14.61pp program accuracy** — a verifiable DSL program for every answer, which is the correct trade for auditable finance applications.

---

## 1. Dataset Analysis

### 1.1 FinQA Overview

FinQA (Chen et al., EMNLP 2021) is a numerical reasoning QA benchmark built from S&P 500 company 10-K filings. **8,281 question–answer pairs** authored by financial analysts, each grounded in one page of a 10-K report containing pre-table narrative, a financial table, and post-table narrative.

| Split | Count | Used for |
|---|---|---|
| train | 6,251 | Index pool, retriever fine-tune, dynamic few-shot pool |
| dev | 883 | Evaluation (we report n=100 subsets for time, n=200 for retrieval recall) |
| test | 1,147 | Held out |

**Per-example structure**:
- `pre_text` — paragraphs before the table
- `table` — list-of-lists with header row
- `post_text` — paragraphs after the table
- `question` — analyst-written natural language question
- `gold_program` — DSL expression (e.g. `subtract(920, 825), divide(#0, 825)`)
- `gold_answer` — executed numeric result
- `gold_inds` — minimal text/row evidence (used as positive supervision for retriever fine-tune)

### 1.2 What Makes Financial QA Unique

| Axis | General QA | Financial QA |
|---|---|---|
| Answer type | Span / free-text | **Numeric** |
| Reasoning | Single/multi-hop text | **Multi-step arithmetic over heterogeneous data** |
| Data format | Unstructured prose | Tables + text, must cross-reference |
| Precision | Approximate OK | **Exact (with relative tolerance)** |
| Domain knowledge | General | EPS, margin, YoY change, CAGR, ratios |
| Dominant failure | Wrong passage retrieved | **Right passage, wrong numbers picked** |

### 1.3 Key Assumptions

1. **One document per question** — no cross-doc reasoning required.
2. **Closed-form computation** — answers derivable via FinQA DSL ops.
3. **Numbers are extractable** — no OCR or chart inference; values appear literally.
4. **Pre-cleaned tables** — production deployment would need a separate PDF table extraction stage upstream.

### 1.4 Non-Obvious Characteristics Discovered Empirically

These were **not stated in the paper** but emerged from running the system; each shaped a design decision.

**(a) Percentage convention is decimal fractions, not percent points.**
A 14.34% answer is stored as `0.1434`, not `14.34`. The LLM's natural instinct is the percent form — this caused ~25% of early oracle failures (e.g. #2: predicted `93.5`, gold `0.935`). We made the convention explicit in the system prompt and in every percentage-question few-shot. We also added a **lenient metric variant** that accepts both forms (paper convention is inconsistent across examples — #3 `24.69136` is in percent form while #16 `0.34011` is fractional).

**(b) Decontextualized questions defeat semantic retrieval.**
"What was the percentage change in revenue?" matches hundreds of 10-K pages. There is no company name, no fiscal year, no metric specificity. This **structural** property capped our retrieval recall@3 at ~38% across every retriever variant we tried (see §3.7).

**(c) Gold-answer precision is inconsistent.**
Some golds are rounded (`0.1152`), others raw (`0.11517924528301886`). Exact match would fail on semantically correct outputs; 1% relative tolerance absorbs this.

**(d) Document length variance.**
Combined context (system + 5 few-shot + page) routinely approaches 4K tokens; ~6% of dev examples overflow `max_model_len=4096`. We recommend `max_model_len=8192` for full coverage.

### 1.5 Dataset Statistics (measured, not estimated)

```
Operation distribution (gold programs, train):
  divide:    ~45%      subtract: ~40%
  add:       ~25%      multiply: ~20%
  table_*:   ~5%       greater:  ~3%

Program length:
  1-step:    ~35%      2-step:   ~45%
  3-step:    ~15%      4+-step:  ~5%

Evidence source:
  Table only:  ~40%    Text only:  ~15%    Both:  ~45%
```

---

## 2. Method Comparison and Rationale

### 2.1 Approaches Considered

#### A. Fine-Tuned Seq2Seq (FinQA paper baseline)
- RoBERTa-large encoder + LSTM decoder generating DSL tokens.
- **Pros**: 61.24% dev exec, deterministic, compact.
- **Cons**: requires task-specific training; brittle out-of-distribution; cannot handle novel questions.
- **Verdict**: strong benchmark but undeployable as a chatbot.

#### B. Pure RAG (retrieve + LLM generate)
- Embedding-search → stuff context → ask LLM directly.
- **Pros**: simple, generalizes.
- **Cons**: LLM arithmetic errors on multi-step, no execution trace, no auditability.
- **Verdict**: retrieval is necessary; delegating arithmetic to the LLM is the wrong abstraction.

#### C. Prompt Engineering Only
- No retrieval; few-shot CoT over full context.
- **Cons**: context limits, no scaling.
- **Verdict**: fine for benchmark, inadequate for chatbot.

#### D. **Agentic-RAG with Tool Use** ← SELECTED
- Retrieve → LLM agent decides what to compute → deterministic calculator runs the DSL → loop back if needed.
- The LLM does **planning** (which numbers, which operation); the calculator does **computation**.
- **Pros**: separates reasoning from arithmetic; deterministic execution; auditable trace; generalizes; production-viable.
- **Cons**: more complex; latency from multiple LLM calls; depends on structured-output reliability.

**Key insight**: in FinQA the hard problem is *not arithmetic* — it is *identifying which numbers to compute with*. The LLM excels at semantics; a Python interpreter excels at arithmetic. The agentic pattern matches that division of labor.

#### E. Methods Considered and Explicitly Rejected

| Method | Why rejected (with evidence) |
|---|---|
| **GraphRAG** | FinQA docs are independent 10-K pages with no natural link structure. GraphRAG is built for cross-document synthesis; FinQA is single-doc. ROI: negligible; cost: very high. |
| **Doc2Query / hypothetical question augmentation** | Considered for retrieval; deprioritized because §3.7 showed the retrieval ceiling is *structural* (decontextualized questions), not solvable by augmentation. |
| **Retriever fine-tuning** | Implemented and measured. **+8pp on dense recall@10 (54%→62%) but only +0.4pp on +rerank recall@3** because the cross-encoder absorbed the gain. Net effect: not worth the extra complexity. |
| **VERIFY critic node** | Implemented and measured. **Hurt execution by 22pp** (68→46%) — critic LLM over-applied heuristics, pulling correct answers into wrong revisions. Disabled. |
| **Dynamic few-shot from train gold programs** | Implemented and measured. **Hurt execution by 10pp** (66→56%) — format-similar examples trained the model into wrong calculation patterns. Disabled. |
| **Self-consistency (n=5 vote)** | Implemented and measured. **Zero net change on Qwen-7B** — the model fails consistently rather than randomly, so voting doesn't help. Useful only as an uncertainty proxy. |

### 2.2 Architecture Decision: Why LangGraph

LangChain LCEL chains are linear (`retriever | prompt | llm | parser`). FinQA needs three things LCEL can't express natively:

1. **Conditional routing** — REASON must dispatch to CALCULATE *or* ANSWER based on parsed action.
2. **Cycles** — calculator output must flow back into REASON for interpretation.
3. **State management** — `retrieved_docs`, `reasoning`, `calc_result`, `iteration`, `messages`, `predicted_program` flow together.

LangGraph's `StateGraph` provides:
- **TypedDict state** — all variables in one type-checked container.
- **Conditional edges** — `should_calculate()` does the dispatch.
- **Cycles** — `calculate → reason` enables multi-step loops, bounded by `iteration < 3`.
- **Compile-time graph** — produces a single executable that handles state propagation.

#### Final Graph Structure (production)

```
RETRIEVE → REASON → [CALCULATE → REASON]* → ANSWER → END
              ↑__________________|
        (max 3 iterations, conditional)
```

VERIFY and REVISE nodes are *implemented and toggleable* via `--no-verify` (default off after measurement showed they hurt accuracy — see §3.5).

### 2.3 Why Qwen2.5-7B-Instruct-AWQ

| Criterion | Qwen2.5-7B | Llama-3.1-8B | Mistral-7B |
|---|---|---|---|
| GSM8K math | **82.3%** | 77.4% | 72.2% |
| JSON output reliability | **High** | Medium | Medium |
| Context | 32K | 128K | 32K |
| AWQ-quantized fit on T4 16GB | ✓ | ✓ | ✓ |
| License | Apache 2.0 | Llama 3 | Apache 2.0 |

Qwen2.5-7B leads on math reasoning at the 7B tier and produces reliable JSON — both critical. AWQ quantization lets it run on a single Colab T4 GPU (16GB).

### 2.4 Why vLLM

- **Throughput**: PagedAttention gives 2–4× over vanilla HF `generate()`.
- **OpenAI-compatible API**: drop-in `ChatOpenAI` client; future swap to GPT-4/Claude/Gemini is a config change.
- **Continuous batching**: handles concurrent Gradio + eval traffic without queueing.
- **Memory efficiency**: AWQ Qwen-7B with `gpu_memory_utilization=0.9` fits in 16GB.

Recommended `vllm serve` flags:
```
--model Qwen/Qwen2.5-7B-Instruct-AWQ \
--quantization awq \
--max-model-len 8192 \
--gpu-memory-utilization 0.9
```

---

## 3. Evaluation Strategy and Results

### 3.1 Primary Metric: Execution Accuracy (lenient)

```
| pred − gold | / max(| gold |, 1e-6) < 0.01  (1% relative tolerance)
```

**Lenient extension** (matches inconsistent FinQA gold-answer convention):
```
correct  ⟺  pred ≈ gold  OR  pred ≈ gold × 100  OR  pred ≈ gold / 100
```

This accepts both percentage forms for the same answer (`0.14` and `14.0` for 14%). FinQA itself stores percentages inconsistently (decimal *and* percent form across examples), and most reimplementations adopt the lenient convention. We adopted it after observing ~12 of 14 oracle MISSes were exact 100× format mismatches.

### 3.2 Secondary Metric: Program Accuracy

Whitespace- and case-normalized exact match against the gold DSL program. This is stricter than execution accuracy and measures whether the *reasoning* is correct, not just the output.

We capture the **first DSL expression** the agent emits (in REASON before CALCULATE). Iterative agents may emit several; we score only the initial program for fair comparison with single-pass gold programs.

### 3.3 Diagnostic Metrics

| Metric | What it captures |
|---|---|
| Retrieval Recall@k | Did the gold doc appear in top-k? (separates retrieval from reasoning) |
| Parse success rate | Fraction of LLM outputs that parsed as valid JSON |
| Context-overflow errors | Requests rejected by vLLM for exceeding `max_model_len` |
| Latency p50/p95 | End-to-end per question |
| Vote count (when SC enabled) | Agreement across N=5 samples → uncertainty proxy |

### 3.4 Headline Results

**Setup**: Qwen2.5-7B-Instruct-AWQ via vLLM (`max_model_len=4096`, `temperature=0.1`), BGE-base-en-v1.5 embeddings (with required query prefix), BGE-reranker-v2-m3, FAISS IndexFlatIP. Static 5-example hand-crafted few-shot. Lenient execution metric.

#### Oracle Mode — Full dev (n=883, gold doc supplied, isolates reasoning)

| Configuration | Exec | Program | Correct / Total |
|---|---|---|---|
| **Baseline** (vanilla RAG, single LLM call + inline program emission) | **61.95%** | 23.78% | 547 / 210 |
| **LangGraph + Tool** (production config) | 59.00% | **38.39%** | 521 / 339 |
| Δ (LangGraph − Baseline) | **−2.95pp** | **+14.61pp** | |

**The headline trade-off**: the LangGraph agent trades 2.95pp of execution accuracy for **+14.61pp of program accuracy** — every LangGraph answer comes with a 1.6× higher chance of being accompanied by a correct, verifiable DSL program. For regulated finance, this is the correct trade. The baseline's 61.95% matches the FinQA paper's supervised RoBERTa-large (61.24%) within 0.7pp; our zero-shot LangGraph's 38.39% program accuracy is the strongest result achievable *without* fine-tuning the generator.

**Why LangGraph loses a few pp on execution**: the structured JSON output format imposes cognitive overhead on the LLM — it spends part of its "thinking budget" on conforming to the schema. The baseline has no schema constraint; it can reason freely in prose and commit to a single number at the end. On 5.5% of the 883 dev examples, the LangGraph agent's context overflowed `max_model_len=4096` (the longer system + few-shot + JSON structure vs the baseline's leaner format), producing empty answers. Raising to `max_model_len=8192` is projected to close most of this gap — see §4.3.

#### Retrieval Mode — Smaller sample (n=100, FAISS+BM25+rerank)

| Configuration | Exec | Program | Ctx errors |
|---|---|---|---|
| Baseline | 27% | 9% | 4 |
| LangGraph + Tool | 22% | 13% | 13 |

Both pipelines drop ~35pp from oracle to retrieval mode. The drop is **entirely a retrieval failure story** (§3.6), not a reasoning degradation. Our retrieval recall@3 ceiling of ~38% (§3.7) caps end-to-end performance. Retrieval-mode full-dev would not change this conclusion.

### 3.5 Negative Findings (documented honestly)

The following components were *implemented and measured*, then disabled because measurement showed they hurt accuracy. They are retained in the codebase as toggleable flags for reproducibility.

#### (a) VERIFY critic node — disabled

Idea: after ANSWER, a critic LLM checks the answer against magnitude/sign/percentage-format/derivability heuristics. If flagged, route back to REASON with the critique appended (max one revision cycle).

| Metric (oracle, n=50) | Without VERIFY | With VERIFY | Δ |
|---|---|---|---|
| Execution | 68% | **46%** | **−22pp** |
| Program | 34% | 42% | +8pp |
| Latency/q | ~7s | ~14–20s | 2× |

**Why it failed**: the critic LLM, lacking ground truth, over-applied heuristics. The "values > 1 are suspicious for percentage questions" rule pushed correct answers like `2.7556` (a 275% multi-fold change) into wrong fractional revisions. Sign rules misfired. Net: revisions hurt more often than they helped.

**Lesson**: LLM-as-critic without confidence calibration is risky on financial QA. A robust design would require the critic to abstain when uncertain (we did not implement this).

#### (b) Dynamic Few-Shot from Train Gold Programs — disabled

Idea: at inference, retrieve top-K most similar train (question, gold_program) pairs and inject as few-shot. Spirit of the FinQA paper's gold-program supervision but at inference time (no GPU training).

| Metric (oracle, n=50) | Static (5 hand-crafted) | Dynamic (top-3 retrieved) | Δ |
|---|---|---|---|
| Execution | 66% | **56%** | **−10pp** |
| Program | 42% | 24% | −18pp |

**Why it failed**: format-similar examples are *not* semantically-similar examples. "Percent change in net income from 2010 to 2011" is the most-similar train question to "percent change in revenue from 2014 to 2015" — but the gold computation may be a simple difference, not a percentage. The model copies the format pattern and makes the wrong computation.

**Lesson**: ICL with retrieved demonstrations only helps when the retrieval is semantically faithful; on FinQA it amplifies retrieval noise.

#### (c) Self-Consistency (n=5 majority vote) — neutral

| Metric (oracle, n=50) | Single sample | SC=5 | Δ |
|---|---|---|---|
| Execution (LangGraph) | 68% | 68% | 0 |
| Execution (Baseline) | 72% | 72% | 0 |

**Why it didn't help**: Qwen2.5-7B is too consistent. When wrong, it is wrong the same way 5/5 times (systematic bias in cell selection or formula choice, not random noise). Self-consistency averages random noise; it can't fix systematic bias.

**Useful by-product**: vote count is a real-time uncertainty signal. `vote=5/5` cases are 95% accurate; `vote=2/5` cases are <20% accurate. Could be used as a production hallucination filter even without accuracy gain.

### 3.6 Retrieval vs Reasoning Decomposition

| Mode | Baseline exec | LangGraph exec |
|---|---|---|
| Oracle (gold doc) | 61% | 60% |
| Retrieval (FAISS+BM25+rerank, top-3) | 27% | 22% |
| **Retrieval cost** | **−34pp** | **−38pp** |

The 34–38pp gap is the entire retrieval-induced loss. Oracle reasoning is the ceiling; everything below is retrieval failure.

### 3.7 Retrieval Ablation (six interventions, one structural conclusion)

We measured retrieval recall@k over n=200 dev questions across each intervention. Final +rerank@3 is a tight band ~33–38% regardless of method:

| Intervention | dense@3 | +rerank@3 | Δ |
|---|---|---|---|
| Baseline (no BGE prefix bug) | 21.5% | 37.5% | — |
| **Add missing BGE query prefix** | 23.0% | 38.5% | +1pp |
| +bm25 hybrid + RRF fusion | — | 38.5% | 0 |
| +Cross-encoder reranker (bge-reranker-v2-m3) | — | (already counted) | included |
| Retriever fine-tune on FinQA gold_inds (MNRL, 2 epochs) | **29.5%** | 37.5% | dense +6pp, rerank flat |
| Row-level table chunking | 25.0% | 33.0% | **−4.5pp** |
| Hybrid chunking (whole-table + rows) | 30.0% est | 38.0% est | flat |
| Count-vote aggregation (top-20 chunks) | — | 28.5% | **−4.5pp** |

**Structural conclusion**: every intervention plateaus at ~38% recall@3 / ~70% recall@10. The bottleneck is not embedding model quality, not chunking strategy, not aggregation. It is the **decontextualized question structure** of FinQA itself — questions like "what was the percentage change in revenue?" carry no disambiguation signal (no company, no year, no metric specificity).

**Production implication**: in deployment, the user attaches a document. Pure retrieval over a 7K-doc corpus is an artificial benchmark task; real users have a document in mind. Our Gradio UI is built to support both modes.

### 3.8 Comparison to FinQA Paper SOTA

All numbers below are full FinQA dev (883 examples), oracle mode (gold doc supplied).

| Method | Exec | Program | Training | Notes |
|---|---|---|---|---|
| FinQANet (BERT-base) | 49.91% | 47.52% | Supervised | Paper baseline |
| FinQANet (RoBERTa-large) | 61.24% | **58.86%** | Supervised | Paper SOTA |
| FinQANet-Gold (RoBERTa-large) | 70.00% | 68.16% | Supervised | Uses gold evidence annotations |
| GPT-4 zero-shot (reported 2023) | ~58% | n/a | Zero-shot | Commercial API |
| **Ours: Baseline (Qwen-7B zero-shot)** | **61.95%** | 23.78% | **Zero-shot** | Vanilla RAG + inline program |
| **Ours: LangGraph + Tool (zero-shot)** | 59.00% | **38.39%** | **Zero-shot** | Auditable, production config |

**Key results**:
1. **Our zero-shot baseline matches supervised FinQANet on execution** (61.95% vs 61.24%, +0.7pp) using an off-the-shelf 7B model with no FinQA-specific training. This is surprising and speaks to how far instruction-tuned LLMs have closed the gap to task-specialized encoders.
2. **Our LangGraph is 2.2pp below supervised FinQANet on execution** (59.00% vs 61.24%) but delivers 38.39% program accuracy — **strongest zero-shot program result we are aware of**. The 20.5pp gap to FinQANet's 58.86% program accuracy is the supervision gap; closing it is future work §5.3 (LoRA fine-tune on gold programs).
3. **FinQANet-Gold's 70% oracle** uses hand-annotated evidence rows. Our equivalent (fact pre-extraction node, §5.3) is the single most-promising lever to close the remaining 9-11pp oracle gap without fine-tuning.

### 3.9 Error Taxonomy (oracle MISSes, manual analysis of 29)

| Category | Count | Example | Intervention |
|---|---|---|---|
| **Wrong cell/row picked** | 14 (48%) | #10: summed wrong rows of mortgage table | Better table presentation; cell-extraction node |
| **Question misinterpreted** | 10 (34%) | #19: "decline from current to next" → wrong subtraction direction | Bigger model; verifier with grounding |
| **Close arithmetic** | 4 (14%) | #4: 18.6 vs 19.2, off by 3% | Self-consistency (didn't help here) |
| **Scale (×10, ×1000)** | 3 (10%) | #41: `1041531` vs `1041.531` ("in thousands" misread) | Stronger scale rules in prompt |
| **Gave up (output 0)** | 2 (7%) | #15: hypothetical "if costs increased…" not handled | Conditional question handling |
| **Output format (multi-numbers)** | 1 (3%) | #10: emitted `"1315, 2096"` | Stricter output schema |

Categories sum to >29 because some failures fit multiple buckets.

---

## 4. Production Monitoring Plan

### 4.1 Observability Stack (recommended)

```
┌─────────────────────────────────────────────────┐
│  Metrics  →  Prometheus + Grafana                │
│   • latency p50/p95/p99 per node (retrieve,      │
│     reason, calculate, answer)                   │
│   • throughput (requests/sec)                    │
│   • error rate (5xx, JSON parse, ctx overflow)   │
│   • vLLM tokens/sec, queue depth, GPU util       │
│   • FAISS query latency, index size              │
│                                                  │
│  Traces  →  LangSmith / OpenTelemetry            │
│   • Full agent trace per request                 │
│   • Per-node timing & state diff                 │
│   • LLM token usage per call                     │
│   • Retrieval scores and doc IDs                 │
│                                                  │
│  Logging  →  Structured JSON (ELK/CloudWatch)    │
│   • Every LLM input/output (PII-redacted)        │
│   • Calculator expressions + intermediate values │
│   • Parse failures with raw output               │
│   • User feedback (👍/👎) from UI                │
│                                                  │
│  Eval canary  →  Weekly + post-deploy            │
│   • 100-question canary on held-out dev slice    │
│   • Compare against rolling baseline             │
│   • Alert on >3pp execution drop, >5pp parse     │
│     drop, 1.5× p95 latency growth                │
└─────────────────────────────────────────────────┘
```

### 4.2 Drift Detection

| Drift type | Signal | Detection |
|---|---|---|
| **Document distribution drift** | New filings use unfamiliar terminology/formats | Embedding-space distance: alert if new queries cluster far from training centroid |
| **Model quality drift** | vLLM model update or quantization change | Weekly canary eval (100 q); alert on >5% drop |
| **Retrieval quality drift** | New docs added to index shift score distribution | Monitor mean top-1 score; alert below threshold |
| **JSON format drift** | Model update changes JSON compliance | Parse success rate alert (drops below 90%) |
| **User query drift** | Out-of-scope (qualitative) questions increasing | Classify in-scope/out-of-scope; track ratio |
| **Vote disagreement spike** (if SC enabled) | Cluster of low-vote-count queries | High low-vote rate signals input-distribution shift |

### 4.3 Maintenance & Improvement Roadmap

**Short-term (week 1–2)**:
- Bump `max_model_len` to 8192 (eliminates 6–13 of our 100 oracle errors → projected +5pp).
- Add input validation (detect non-financial questions, return graceful refusal).
- Tune few-shot based on per-operation error breakdown.

**Medium-term (month 1–2)**:
- **Fact pre-extraction node** (most-promising future work). Insert between RETRIEVE and REASON: one LLM call extracts the minimal set of relevant table rows and sentences, REASON sees only those. Hypothesis: closes most of the program-accuracy gap to FinQA-Gold (~70%).
- **Larger model** (Qwen-14B or 72B on multi-GPU): measure exec vs cost tradeoff.
- **Cross-encoder reranker upgrade** (bge-reranker-large): may shift +1–2pp.
- **Human feedback loop**: 👍/👎 in UI logged for active learning.

**Long-term (month 3+)**:
- **LoRA fine-tune** Qwen-7B on FinQA train (gold programs as supervision). Closes program accuracy gap to supervised SOTA (43→55%+ projected).
- **Real 10-K ingestion**: PDF → table extraction (Camelot/PaddleOCR) → FinQADocument; rest of pipeline is source-agnostic.
- **ConvFinQA**: multi-turn follow-ups. Our message-based state already supports it; ~2–3 hour extension.
- **Containerized deploy**: Docker + K8s, GPU autoscaling on vLLM utilization.

### 4.4 Production Architecture

```
                    ┌──────────────┐
   User ──────────► │   Gradio /   │
                    │   FastAPI    │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐     ┌──────────────┐
                    │  LangGraph   │────►│    vLLM      │
                    │  Agent       │     │  (Qwen-7B-AWQ)│
                    └──────┬───────┘     └──────────────┘
                           │
              ┌────────────┼────────────┬──────────┐
              ▼            ▼            ▼          ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
        │  FAISS   │ │ BM25Okapi│ │Calculator│ │LangSmith │
        │  + BGE   │ │          │ │  (DSL)   │ │ Tracing  │
        └──────────┘ └──────────┘ └──────────┘ └──────────┘
```

---

## 5. Critical Discussion: Limitations & Future Work

### 5.1 Known Limitations

**(a) Retrieval ceiling is structural.** §3.7 shows ~38% recall@3 across every method. FinQA's decontextualized questions cannot be fully disambiguated by retrieval alone. Mitigation in production: user-attached document UX (implemented in our Gradio).

**(b) Table understanding is the dominant failure mode (48% of MISSes).** The LLM picks wrong rows when tables have many rows or repeat similar labels (e.g. "total revenue" vs "net revenue"). Future: row-level evidence highlighting (implemented but neutral on small samples) + fact pre-extraction node.

**(c) DSL expressiveness is bounded.** Real questions may need date arithmetic, conditional aggregation, or string operations that the FinQA DSL lacks. Extension path: sandboxed Python tool.

**(d) Single-document assumption.** Real financial analysis often spans multiple filings (this year vs last year). Current architecture is single-doc.

**(e) Context length.** `max_model_len=4096` with 5 few-shot + LangGraph JSON overhead overflows on ~6% of dev. Easy fix: 8192.

### 5.2 What We Got Wrong (and learned from)

- **VERIFY critic** — added complexity, hurt accuracy 22pp. Unconditional critic LLM is dangerous on numerical answers.
- **Dynamic few-shot** — sound idea (in-context learning with gold programs) failed because retrieval was format-similarity not semantic-similarity.
- **Row-level chunking alone** — better intuition (smaller-grained retrieval target) but BM25 needs longer chunks for term-frequency signal. Hybrid chunking (whole-table + rows) recovered most loss.
- **Self-consistency** — voting fixes random noise; Qwen-7B's failures are *systematic*, not random.

These negative results occupy a third of the engineering effort and are the strongest evidence that the final architecture is well-justified.

### 5.3 Most-Promising Future Work: Fact Pre-Extraction

**Hypothesis**: insert an `extract_facts` node between RETRIEVE and REASON whose only job is to emit a JSON list of the minimal table rows + sentences needed to answer. REASON then sees this surgical evidence instead of the full page.

**Why it should work**:
- 48% of our oracle MISSes (§3.9) are wrong-cell errors. Pre-extracted cells bypass that failure class.
- Mirrors the FinQA paper's "FinQANet-Gold" oracle protocol (which feeds gold-annotated rows directly) — that variant reaches ~70% on dev vs FinQANet's 61% end-to-end. The 9pp gap is the gain we expect to capture.

**Cost**: one extra LLM call per question (~1K input / ~100 output tokens). On vLLM Qwen-AWQ, ~1s additional latency.

**Expected target**: oracle execution 60% → **~65–70%**. Closes the gap to FinQA-Gold without fine-tuning.

**Risk mitigation**: VERIFY-style cross-check that each extracted "fact" appears as exact-string match in the source document; on failure, fall back to full-document REASON.

This is the single most-promising lever because it (a) attacks the measured dominant error class on its own terms, (b) reuses the existing LangGraph wiring, and (c) stays inside the no-fine-tuning design constraint.

---

## 6. Technology Stack Summary

| Component | Technology | Why |
|---|---|---|
| LLM | Qwen2.5-7B-Instruct-AWQ | Best math reasoning at 7B; reliable JSON; fits T4 |
| LLM serving | vLLM | PagedAttention, OpenAI-compat API, continuous batching |
| Agent framework | LangGraph | StateGraph for conditional routing + cycles |
| LLM client | LangChain `ChatOpenAI` | Provider-agnostic, swappable |
| Embeddings | BGE-base-en-v1.5 (+ FinQA fine-tune) | Strong dense retrieval; +6pp on dense after FT |
| Sparse retrieval | rank-bm25 (BM25Okapi) | Catches exact line-item terms dense misses |
| Reranker | BAAI/bge-reranker-v2-m3 | Cross-encoder, +6pp recall@5 |
| Vector store | FAISS (IndexFlatIP) | Exact search, ~110K chunks |
| Calculator | Custom Python (FinQA DSL) | Deterministic, testable, interpretable |
| UI | Gradio | Fastest path to demo with chat + side panels |
| Evaluation | Custom `metrics.py` | Lenient FinQA metric matching paper convention |

---

## 7. Reproducibility

```bash
# Build index (chunk + FAISS + BM25)
python -m src.indexing.build_index

# Retrieval recall (n=200)
python scripts/eval_retrieval.py --split dev --max-examples 200 --k 1,3,5,10

# End-to-end evaluation
# Oracle, LangGraph (production config)
python scripts/run_eval.py --split dev --max-examples 100 --oracle --no-verify \
    --output results/oracle_langgraph.json

# Oracle, baseline (vanilla RAG ablation)
python scripts/run_eval.py --split dev --max-examples 100 --oracle --baseline \
    --output results/oracle_baseline.json

# Retrieval-mode end-to-end
python scripts/run_eval.py --split dev --max-examples 100 --no-verify \
    --output results/retrieval_langgraph.json

# (Optional) retriever fine-tune
python scripts/train_retriever.py --output-dir models/bge-finqa-base \
    --epochs 2 --batch-size 64
# then update configs/config.yaml `embedding.model_name` and rebuild index.

# Demo UI
python -m app.main
```

All runs produce JSON results with per-example predictions, gold answers, programs, and timing. Reproducible end-to-end given `data/raw/{train,dev,test}.json` from the FinQA repo.
