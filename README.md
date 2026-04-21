# FinQA Chatbot

An agentic-RAG chatbot for numerical reasoning over financial documents, built on the [FinQA dataset](https://github.com/czyssrs/FinQA).

## Architecture

```
User Question → RETRIEVE (FAISS + BGE) → REASON (Qwen2.5-7B via vLLM) → CALCULATE (DSL) ↺ → ANSWER
```

**Key design**: The LLM handles semantic understanding (identifying numbers, choosing operations). Arithmetic is delegated to a deterministic calculator, eliminating computation errors entirely.

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Qwen2.5-7B-Instruct |
| LLM Serving | vLLM (OpenAI-compatible API) |
| Agent Framework | LangGraph (StateGraph) |
| LLM Integration | LangChain (ChatOpenAI) |
| Embeddings | BGE-base-en-v1.5 (768-dim) |
| Vector Store | FAISS (IndexFlatIP) |
| UI | Gradio |

## Quick Start

### Prerequisites
- Python 3.10+
- NVIDIA GPU with ≥16GB VRAM (for vLLM)
- ~2GB disk for model downloads

### Setup

```bash
# Install dependencies
pip install -r requirements.txt
pip install vllm  # GPU server only

# Download FinQA dataset
make data

# Build FAISS index (embeds ~7K documents)
make index

# Start vLLM server (separate terminal)
make serve

# Launch chatbot UI
make ui
```

Open http://localhost:7860 in your browser.

### Evaluation

```bash
# Run on dev set (full)
python scripts/run_eval.py --split dev

# Quick test (50 examples)
python scripts/run_eval.py --split dev --max-examples 50

# Error analysis
python -m src.evaluation.analyze results/eval_results.json
```

## Project Structure

```
finqa-chatbot/
├── configs/config.yaml           # All hyperparameters
├── data/download.py              # Dataset downloader
├── src/
│   ├── data_processing/          # Document parsing, table conversion
│   ├── indexing/                 # BGE embeddings, FAISS index
│   ├── retrieval/                # LangChain BaseRetriever
│   ├── tools/                    # FinQA DSL calculator
│   ├── llm/                      # vLLM client factory
│   ├── agent/                    # LangGraph StateGraph (core pipeline)
│   └── evaluation/               # Metrics, batch runner, error analysis
├── scripts/                      # CLI entry points
├── app/                          # Gradio web interface
├── tests/                        # Unit tests
└── REPORT.md                     # Technical report
```

## Documentation

See [REPORT.md](REPORT.md) for:
- Dataset analysis with real statistics
- Method comparison (fine-tuned vs. RAG vs. agentic-RAG)
- Evaluation strategy and metrics
- Production monitoring plan
- Known limitations and improvement roadmap
